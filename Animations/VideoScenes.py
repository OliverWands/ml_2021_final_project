import math
import os
import random

import manim
import pandas as pd
from manim import *
from scipy.signal import medfilt, butter, sosfiltfilt


def get_inx_val(base, data, start, end, axis):
    three_dim_file = pd.read_csv(os.path.join(base, data), sep=" ", header=None)
    return three_dim_file[axis][start:end]


raw = "dataset/RawData"
acc = "acc_exp01_user01.txt"

# Third order lowpass butterworth filter to remove noise.
# cutoff frequency of 20Hz sampling rate 50Hz
noise_filter = butter(btype='lp', N=3, fs=50, Wn=20 / (50 / 2), output='sos', analog=False)

# Third order lowpass butterworth filter to get the gravity component.
# cutoff frequency of 0.3Hz sampling rate 50Hz
grav_filter = butter(btype='lp', N=3, fs=50, Wn=0.3 / (50 / 2), output='sos', analog=False)


def return_coords_from_array(array_data, axes):
    coords = []
    for inx, val in enumerate(array_data):
        coords.append(axes.coords_to_point(float(inx * 0.02), float(val)))
    return coords


def create_line_plot(coordinates, line_color, strokewidth=1.25):
    return VGroup(
        *[Line(start=(coord[0], coord[1], 0.0), end=(coordinates[inx + 1][0], coordinates[inx + 1][1], 0.0),
               stroke_width=strokewidth, color=line_color) for inx, coord in enumerate(coordinates[:-1])])


def preprocess_axis(array_data):
    vals_med = medfilt(array_data)
    rem_noise = sosfiltfilt(noise_filter, vals_med)
    grav = sosfiltfilt(grav_filter, rem_noise)
    vals_body = rem_noise - grav
    vals_jerk = [(vals_body[inx - 1] - vals_body[inx]) / 0.02 for inx in range(1, len(vals_body))]

    return vals_med, rem_noise, grav, vals_body, vals_jerk


def calcluate_mag(x_values, y_values, z_values):
    return np.sqrt(np.square(x_values) + np.square(y_values) + np.square(z_values))


def invisible_dot(axes, x_coord, y_coord):
    return Dot(axes.coords_to_point(float(x_coord), float(y_coord)), radius=0.0001, stroke_opacity=0)


class Preprocessing(Scene):
    config.background_color = WHITE

    def construct(self):
        axes = Axes(
            y_range=[-0.6, 0.6, 0.1],
            x_range=[0, 12.0, 1],
            x_length=9,
            y_length=6,
            x_axis_config={
                "numbers_to_include": [12],
                "numbers_with_elongated_ticks": np.arange(0, 12.01, 2),
            },
            y_axis_config={
                "numbers_to_include": [-0.6, 0, 0.6],
                "numbers_with_elongated_ticks": np.arange(-0.6, 0.6, 0.2),
            },
            tips=False
        )
        axes.set_color(GRAY_E)
        labels = axes.get_axis_labels(x_label="", y_label="")

        z_vals_raw = get_inx_val(raw, acc, 7496, 8078, 2).tolist()
        z_vals_med, z_rem_noise, z_grav, z_vals_body, _ = preprocess_axis(z_vals_raw)

        raw_lines = create_line_plot(return_coords_from_array(z_vals_raw, axes), RED_E, strokewidth=1)
        med_lines = create_line_plot(return_coords_from_array(z_vals_med, axes), GOLD_E, strokewidth=1)
        noise_lines = create_line_plot(return_coords_from_array(z_rem_noise, axes), YELLOW_E, strokewidth=1)
        but_lines = create_line_plot(return_coords_from_array(z_vals_body, axes), BLUE_E, strokewidth=1)

        grav_coords = return_coords_from_array(z_grav, axes)
        grav_lines = create_line_plot(grav_coords, TEAL_E, strokewidth=1.25)
        grav_to_zero = VGroup(*[Line(start=(coord[0], 0.0, 0.0), end=(grav_coords[inx + 1][0], 0.0, 0.0),
                                     stroke_width=1.25, color=BLUE_E) for inx, coord in enumerate(grav_coords[:-1])])

        dot = invisible_dot(axes, 10, 0.5)

        raw_text = Tex("raw", color=RED_E).scale(1)
        raw_text.next_to(dot, RIGHT)

        med_text = Tex("medfilt(raw)", color=GOLD_E).scale(1)
        med_text.next_to(dot, RIGHT)

        noise_text = Tex("noise removed", color=YELLOW_E).scale(1)
        noise_text.next_to(noise_lines, UP)

        grav_text = Tex("gravity acceleration", color=TEAL_E).scale(1)
        grav_text.next_to(grav_lines, DOWN)

        body_text = Tex("body acceleration", color=BLUE_E).scale(1)
        body_text.next_to(but_lines, UP)

        x_label_text = Tex("time (sec)", color=GRAY_E).scale(0.5)
        x_label_text.next_to(invisible_dot(axes, 12.25, 0.05), UP)
        y_label_text_up = Tex("acceleration", color=GRAY_E).scale(0.5)
        y_label_text_up.next_to(invisible_dot(axes, 0.25, 0.6), UR)
        y_label_text_down = Tex("($g \\rightarrow 9.80665 \\frac{m}{seg^2}$)", color=GRAY_E).scale(0.4)
        y_label_text_down.next_to(y_label_text_up, DOWN)

        grav_lines_2 = grav_lines.copy()

        self.add(axes, labels, dot, x_label_text, y_label_text_up, y_label_text_down)
        self.play(FadeIn(raw_lines), FadeIn(raw_text))
        self.wait()
        self.play(Transform(raw_lines, med_lines, replace_mobject_with_target_in_scene=True),
                  FadeOut(raw_text),
                  FadeIn(med_text))
        self.wait(3)
        self.play(Transform(med_lines, noise_lines, replace_mobject_with_target_in_scene=True),
                  Transform(med_text, noise_text, replace_mobject_with_target_in_scene=True))
        self.wait(9)
        self.play(FadeIn(grav_lines), FadeIn(grav_text))
        self.wait(10)
        self.play(Transform(noise_lines, but_lines, replace_mobject_with_target_in_scene=True),
                  Transform(grav_lines, grav_to_zero, replace_mobject_with_target_in_scene=True),
                  Transform(noise_text, body_text, replace_mobject_with_target_in_scene=True))
        self.wait(2)
        self.play(Transform(grav_to_zero, grav_lines_2, replace_mobject_with_target_in_scene=True))
        self.wait(3)
        self.play(*[FadeOut(mob) for mob in self.mobjects])


class SlidingWindow(MovingCameraScene):
    config.background_color = WHITE

    def construct(self):
        axes = Axes(
            y_range=[-0.4, 0.6, 0.1],
            x_range=[0, 12.0, 1],
            x_length=9,
            y_length=6,
            x_axis_config={
                "numbers_to_include": [12],
                "numbers_with_elongated_ticks": np.arange(0, 12.01, 2),
            },
            y_axis_config={
                "numbers_to_include": [-0.4, 0, 0.6]
            },
            tips=False
        )

        axes.set_color(GRAY_E)

        x_vals_raw = get_inx_val(raw, acc, 7496, 8078, 0).tolist()
        x_vals_med, x_rem_noise, x_grav, x_vals_body, _ = preprocess_axis(x_vals_raw)
        y_vals_raw = get_inx_val(raw, acc, 7496, 8078, 1).tolist()
        y_vals_med, y_rem_noise, y_grav, y_vals_body, _ = preprocess_axis(y_vals_raw)
        z_vals_raw = get_inx_val(raw, acc, 7496, 8078, 2).tolist()
        z_vals_med, z_rem_noise, z_grav, z_vals_body, _ = preprocess_axis(z_vals_raw)

        x_body_line = create_line_plot(return_coords_from_array(x_vals_body, axes), RED, strokewidth=1.0)
        x_grav_line = create_line_plot(return_coords_from_array(x_grav, axes), RED_D)

        y_body_line = create_line_plot(return_coords_from_array(y_vals_body, axes), GREEN, strokewidth=1.0)
        y_grav_line = create_line_plot(return_coords_from_array(y_grav, axes), GREEN_D)

        z_body_line = create_line_plot(return_coords_from_array(z_vals_body, axes), BLUE_E, strokewidth=1.0)
        z_grav_line = create_line_plot(return_coords_from_array(z_grav, axes), BLUE_E)

        windows = list()
        count = 0
        colors = [GREEN_D, GREEN]

        for inx in range(0, 583, 63):
            start = inx
            end = inx + 128
            if end > 583:
                break

            top = 0.55
            bot = -0.3
            left = Line(start=axes.coords_to_point(float(start * 0.02), float(top)),
                        end=axes.coords_to_point(float(start * 0.02), float(bot)),
                        color=colors[count % 2])
            right = Line(start=axes.coords_to_point(float(end * 0.02), float(top)),
                         end=axes.coords_to_point(float(end * 0.02), float(bot)),
                         color=colors[count % 2])
            top_line = Line(start=axes.coords_to_point(start * 0.02, float(top)),
                            end=axes.coords_to_point(float(end * 0.02), float(bot)),
                            stroke_width=0.00001, stroke_opacity=0)
            brace = Brace(top_line, direction=DOWN, color=GRAY_E)
            brace_text = brace.get_text(str(count + 1)).scale(0.75)
            brace_text.set_color(GRAY_E)
            windows.append([left, right, brace, brace_text])
            count += 1

        # grav_text = Tex("gravity acceleration", color=TEAL_E).scale(1)
        # grav_text.next_to(z_grav_line, DOWN)
        # body_text = Tex("body acceleration", color=BLUE_E).scale(1)
        # body_text.next_to(z_body_line, UP)

        first_window_text_1 = Tex(r"128 samples", color=GRAY_E).scale(0.5)
        first_window_text_1.next_to(windows[0][2], DOWN)
        first_window_text_2 = Tex(r"2.56 sec", color=GRAY_E).scale(0.5)
        first_window_text_2.next_to(first_window_text_1, DOWN)

        self.add(axes)
        self.wait(0.5)
        self.play(FadeIn(z_grav_line), FadeIn(z_body_line),
                  FadeIn(y_grav_line), FadeIn(y_body_line),
                  FadeIn(x_body_line), FadeIn(x_grav_line))
        # FadeIn(grav_text), FadeIn(body_text))
        self.wait()
        first_left = windows[1][0].copy()
        first_right = windows[1][1].copy()
        self.play(FadeIn(windows[0][0]), FadeIn(windows[0][1]))
        self.wait()
        self.play(FadeIn(windows[0][2]), FadeIn(first_window_text_1), FadeIn(first_window_text_2))

        self.wait(2)
        for jnx, window in enumerate(windows):
            if jnx == 0:
                self.play(FadeIn(windows[0][3]),
                          # FadeOut(grav_text), FadeOut(body_text),
                          FadeOut(first_window_text_1), FadeOut(first_window_text_2))
                continue
            self.play(
                Transform(windows[jnx - 1][0], window[0], replace_mobject_with_target_in_scene=True, run_time=1.28),
                Transform(windows[jnx - 1][1], window[1], replace_mobject_with_target_in_scene=True, run_time=1.28),
                Transform(windows[jnx - 1][2], window[2], replace_mobject_with_target_in_scene=True, run_time=1.28),
                Transform(windows[jnx - 1][3], window[3], replace_mobject_with_target_in_scene=True, run_time=1.28))
        self.play(FadeOut(windows[len(windows) - 1][0]), FadeOut(windows[len(windows) - 1][1]),
                  FadeOut(windows[len(windows) - 1][2]), FadeOut(windows[len(windows) - 1][3]))
        self.wait()

        mid_dot = Dot(axes.coords_to_point(1.92, 0.1), radius=0.0001, stroke_opacity=0)

        self.play(self.camera.frame.animate.scale(0.9).move_to(mid_dot),
                  FadeIn(first_left), FadeIn(first_right))
        self.wait(5)
        self.play(*[FadeOut(mob) for mob in self.mobjects])


class ShowJerk(Scene):
    def construct(self):
        axes = Axes(
            y_range=[-0.5, 0.5, 0.1],
            x_range=[0, 12.0, 1],
            x_length=9,
            y_length=6,
            x_axis_config={
                "numbers_to_include": [12],
                "numbers_with_elongated_ticks": np.arange(0, 12.01, 2),
            },
            y_axis_config={
                "numbers_to_include": [-0.5, 0, 0.5],
                "numbers_with_elongated_ticks": np.arange(-0.51, 0.51, 0.2),
            },
            tips=False
        )
        axes.set_color(GRAY_E)

        z_vals_raw = get_inx_val(raw, acc, 7496, 8078, 2).tolist()
        z_vals_med, z_rem_noise, z_grav, z_vals_body, z_vals_jerk = preprocess_axis(z_vals_raw)

        body_line = create_line_plot(return_coords_from_array(z_vals_body, axes), BLUE_E, strokewidth=1)
        jerk_line = create_line_plot(return_coords_from_array(z_vals_jerk, axes), GOLD_E, strokewidth=1)
        self.add(axes, body_line)
        self.play(Transform(body_line, jerk_line, replace_mobject_with_target_in_scene=True))
        self.wait(2)


class Opening(Scene):
    def construct(self):
        ml = Text("Machine Learning 2021", color=GRAY_E)
        sub = Text("Final project video submission", color=GRAY_E).scale(0.75)
        title = Text("Human Activity Recognition", color=BLUE_E).scale(1.25)
        name = Text("Oliver Wandschneider", color=TEAL_E).scale(0.75)
        # title_bigger = Text("Human Activity Recognition", color=BLUE_E).scale(1.4)
        # title_bigger.next_to(ORIGIN)

        ml.next_to([0, 2, 0], DOWN)
        sub.next_to(ml, DOWN)
        title.next_to(sub, DOWN)
        name.next_to(title, DOWN)
        self.play(FadeIn(ml), FadeIn(sub), FadeIn(title), FadeIn(name))
        self.wait(4)
        self.play(FadeOut(ml), FadeOut(sub), FadeOut(name), FadeOut(title))
        # Transform(title, title_bigger, replace_mobject_with_target_in_scene=True))
        self.wait(0.5)


class DataDistribution(Scene):
    def construct(self):
        axes = Axes(
            y_range=[0, 2100, 250],
            x_range=[0, 12.9, 1],
            x_length=9,
            y_length=6,
            x_axis_config={
                "numbers_to_include": np.arange(1, 12.01, 1)
            },
            y_axis_config={
                "numbers_to_include": np.arange(0, 2001, 250)
            },
            tips=False
        )

        axes.set_color(GRAY_E)

        def create_bar(height, position, label, color, label_color=GRAY_E, start=0):
            coords_left = []
            for inx in range(start, height):
                coords_left.append(axes.coords_to_point(float(position - 0.35), float(inx)))
            coords_right = []
            for inx in range(start, height):
                coords_right.append(axes.coords_to_point(float(position + 0.35), float(inx)))

            bar = VGroup(
                *[Line(start=(coords_left[inx][0], coords_left[inx][1], 0.0),
                       end=(coords_right[inx][0], coords_right[inx][1], 0.0),
                       stroke_width=1, color=color) for inx in range(start, height)])

            text = Tex(label, color=label_color).scale(0.5)
            text.next_to(Dot(axes.coords_to_point(position, 350), radius=0.0001, stroke_opacity=0), UP)
            text.rotate(np.pi / 2)

            return bar, text

        walking_bar, w_t = create_bar(1722, 1, "Walking", BLUE_E)
        walking_up_bar, w_u_t = create_bar(1544, 2, "Walking upstairs", BLUE_E)
        walking_down_bar, w_d_t = create_bar(1407, 3, "Walking downstairs", BLUE_E)
        sitting_bar, sit_t = create_bar(1801, 4, "Sitting", BLUE_E)
        standing_bar, sta_t = create_bar(1979, 5, "Standing", BLUE_E)
        laying_bar, lay_t = create_bar(1958, 6, "Laying", BLUE_E)

        stand_sit_bar, st_sit_t = create_bar(70, 7, "Stand to sit", TEAL_E)
        sit_stand_bar, sit_st_t = create_bar(33, 8, "Sit to stand", TEAL_E)
        sit_lie_bar, sit_lie_t = create_bar(107, 9, "Sit to lie", TEAL_E)
        lie_sit_bar, lie_sit_t = create_bar(85, 10, "Lie to sit", TEAL_E)
        stand_lie_bar, st_lie_t = create_bar(139, 11, "Stand to lie", TEAL_E)
        lie_stand_bar, lie_st_t = create_bar(84, 12, "Lie to stand", TEAL_E)

        self.add(axes)
        self.play(FadeIn(walking_bar), FadeIn(walking_up_bar), FadeIn(walking_down_bar),
                  FadeIn(sitting_bar), FadeIn(standing_bar), FadeIn(laying_bar),
                  FadeIn(w_t), FadeIn(w_u_t), FadeIn(w_d_t),
                  FadeIn(sit_t), FadeIn(sta_t), FadeIn(lay_t))
        self.wait(2)
        self.play(FadeIn(sit_stand_bar), FadeIn(stand_sit_bar), FadeIn(sit_lie_bar),
                  FadeIn(lie_sit_bar), FadeIn(stand_lie_bar), FadeIn(lie_stand_bar),
                  FadeIn(st_sit_t), FadeIn(sit_st_t), FadeIn(sit_lie_t),
                  FadeIn(lie_sit_t), FadeIn(st_lie_t), FadeIn(lie_st_t))
        self.wait(3)


def create_neuron_layer(n, circle_color, start_origin, dots=True, stroke_width=2, rad=0.3, dropout=0.0):
    elements = [Circle()] * n

    if dots:
        center = Circle(radius=rad, stroke_opacity=0)
    else:
        if random.random() < dropout:
            center = Circle(radius=rad, color=GRAY_D, fill_color=GRAY, fill_opacity=0.5, stroke_width=stroke_width)
        else:
            center = Circle(radius=rad, color=circle_color, stroke_width=stroke_width)

    center.next_to(start_origin, ORIGIN)

    elements[int(math.floor(n / 2))] = center

    for inx in range(math.floor(n / 2) - 1, -1, -1):
        if random.random() < dropout:
            above = Circle(radius=rad, color=GRAY_D, fill_color=GRAY, fill_opacity=0.5, stroke_width=stroke_width)
        else:
            above = Circle(radius=rad, color=circle_color, stroke_width=stroke_width)
        above.next_to(elements[inx + 1], UP)

        elements[inx] = above

    for inx in range(math.floor(n / 2) + 1, n):
        if random.random() < dropout:
            below = Circle(radius=rad, color=GRAY_D, fill_color=GRAY, fill_opacity=0.5, stroke_width=stroke_width)
        else:
            below = Circle(radius=rad, color=circle_color, stroke_width=stroke_width)
        below.next_to(elements[inx - 1], DOWN)
        elements[inx] = below

    if dots:
        dot_text = Tex(r"...", color=GRAY_E).scale(1.25)
        dot_text.next_to(start_origin, ORIGIN)
        elements[int(math.floor(n / 2))] = dot_text

    return elements


def create_dense(layer_1_circles, layer_2_circles, one_to_one=False):
    lines = list()
    for inx, circle_1 in enumerate(layer_1_circles):
        if not isinstance(circle_1, manim.Circle):
            continue
        if str(circle_1.color) == "#444":
            continue
        if one_to_one:
            lines.append(Line(circle_1.get_edge_center(RIGHT), layer_2_circles[inx].get_edge_center(LEFT),
                              color=GRAY, buff=0, stroke_width=0.5))
            continue
        for circle_2 in layer_2_circles:
            if not isinstance(circle_2, manim.Circle):
                continue
            if str(circle_2.color) == "#444":
                continue
            lines.append(Line(circle_1.get_edge_center(RIGHT), circle_2.get_edge_center(LEFT),
                              color=GRAY, buff=0, stroke_width=0.5))

    return VGroup(*lines)


HIDDEN_COL = "#0051A8"
NORM_COL = "#0082A8"
DROP_COL = "#00A89B"


class NeuralNetworkActivity(MovingCameraScene):
    def construct(self):
        # Transitions
        layer_1 = create_neuron_layer(9, RED_E, [-4, 0, 0], True)
        layer_2 = create_neuron_layer(7, HIDDEN_COL, [-2, 0, 0], True)
        layer_norm_1 = create_neuron_layer(7, NORM_COL, [-1, 0, 0], True)
        layer_3 = create_neuron_layer(7, HIDDEN_COL, [1, 0, 0], True)
        layer_norm_2 = create_neuron_layer(7, NORM_COL, [2, 0, 0], True)
        layer_4 = create_neuron_layer(6, GREEN_E, [4, -0.4, 0], False)

        lines_1 = create_dense(layer_1, layer_2)
        lines_2 = create_dense(layer_2, layer_norm_1, one_to_one=True)
        lines_between = create_dense(layer_norm_1, layer_3)
        lines_3 = create_dense(layer_3, layer_norm_2, one_to_one=True)
        lines_4 = create_dense(layer_norm_2, layer_4)

        # Input Layer
        l1_vg = VGroup(*layer_1)
        brace_l1 = Brace(l1_vg, color=GRAY_E, direction=LEFT, sharpness=1.2)
        brace_l1_text = Text("Input size: 561", color=GRAY_E).rotate(np.pi / 2).scale(0.75)
        brace_l1_text.next_to(brace_l1.get_edge_center(LEFT), LEFT)

        # Hidden Layer 1
        l2_vg = VGroup(*layer_2)
        brace_l2 = Brace(l2_vg, color=GRAY_E, direction=UP, sharpness=1.2)
        brace_l2_text_1 = Text("228 Neurons", color=GRAY_E).scale(0.55)
        brace_l2_text_1.next_to(brace_l2, UP)
        brace_l2_text_2 = Text("Sigmoid", color=GRAY_E).scale(0.55)
        brace_l2_text_2.next_to(brace_l2_text_1.get_edge_center(UP), UP, buff=0.1)
        brace_l2_text_3 = Text("Dense", color=GRAY_E).scale(0.55)
        brace_l2_text_3.next_to(brace_l2_text_2.get_edge_center(UP), UP, buff=0.1)

        # Batch Norm 1
        bnorm1_vg = VGroup(*layer_norm_1)
        brace_norm1 = Brace(bnorm1_vg, color=GRAY_E, direction=UP, sharpness=1.2)
        brace_norm1_text_1 = Text("Batch Normalization", color=GRAY_E).scale(0.55)
        brace_norm1_text_1.next_to(brace_norm1, UP)

        # Hidden Layer 2
        l3_vg = VGroup(*layer_3)
        brace_l3 = Brace(l3_vg, color=GRAY_E, direction=UP, sharpness=1.2)
        brace_l3_text_1 = Text("232 Neurons", color=GRAY_E).scale(0.55)
        brace_l3_text_1.next_to(brace_l3, UP)
        brace_l3_text_2 = Text("Sigmoid", color=GRAY_E).scale(0.55)
        brace_l3_text_2.next_to(brace_l3_text_1.get_edge_center(UP), UP, buff=0.1)
        brace_l3_text_3 = Text("Dense", color=GRAY_E).scale(0.55)
        brace_l3_text_3.next_to(brace_l3_text_2.get_edge_center(UP), UP, buff=0.1)

        # Batch Norm 2
        bnorm2_vg = VGroup(*layer_norm_2)
        brace_norm2 = Brace(bnorm2_vg, color=GRAY_E, direction=UP, sharpness=1.2)
        brace_norm2_text_1 = Text("Batch Normalization", color=GRAY_E).scale(0.55)
        brace_norm2_text_1.next_to(brace_norm2, UP)

        # Output Layer
        l4_vg = VGroup(*layer_4)
        brace_l4 = Brace(l4_vg, color=GRAY_E, direction=UP, sharpness=1.2)
        brace_l4_text_1 = Text("6 Neurons", color=GRAY_E).scale(0.55)
        brace_l4_text_1.next_to(brace_l4, UP)
        brace_l4_text_2 = Text("Softmax", color=GRAY_E).scale(0.55)
        brace_l4_text_2.next_to(brace_l4_text_1.get_edge_center(UP), UP, buff=0.1)
        brace_l4_text_3 = Text("Dense", color=GRAY_E).scale(0.55)
        brace_l4_text_3.next_to(brace_l4_text_2.get_edge_center(UP), UP, buff=0.1)

        self.add(lines_1, lines_2, lines_between, lines_3, lines_4,
                 l1_vg, l2_vg, l3_vg, l4_vg,
                 bnorm1_vg, bnorm2_vg)

        self.play(FadeIn(brace_l1), FadeIn(brace_l1_text))
        self.wait(4)
        self.play(self.camera.frame.animate.scale(1.25).move_to(ORIGIN),
                  Transform(brace_l1, brace_l2, replace_mobject_with_target_in_scene=True), FadeOut(brace_l1_text),
                  FadeIn(brace_l2_text_1), FadeIn(brace_l2_text_2), FadeIn(brace_l2_text_3))
        self.wait(6)
        self.play(Transform(brace_l2, brace_norm1, replace_mobject_with_target_in_scene=True),
                  FadeIn(brace_norm1_text_1),
                  FadeOut(brace_l2_text_1), FadeOut(brace_l2_text_2), FadeOut(brace_l2_text_3))
        self.wait(7)
        self.play(Transform(brace_norm1, brace_l3, replace_mobject_with_target_in_scene=True),
                  FadeIn(brace_l3_text_1), FadeIn(brace_l3_text_2), FadeIn(brace_l3_text_3),
                  FadeOut(brace_norm1_text_1))
        self.wait(6)
        self.play(Transform(brace_l3, brace_norm2, replace_mobject_with_target_in_scene=True),
                  FadeIn(brace_norm2_text_1),
                  FadeOut(brace_l3_text_1), FadeOut(brace_l3_text_2), FadeOut(brace_l3_text_3))
        self.wait(4)
        self.play(Transform(brace_norm2, brace_l4, replace_mobject_with_target_in_scene=True),
                  FadeIn(brace_l4_text_1), FadeIn(brace_l4_text_2), FadeIn(brace_l4_text_3),
                  FadeOut(brace_norm2_text_1))
        self.wait(3)
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        one_hot = Text("Example:\none_hot(label=3) = [0, 0, 1, 0, 0, 0]", color="#FF6800")
        self.wait(3)
        self.play(FadeIn(one_hot))
        output = Text("Example:\nOutput of NN = [0, 0.5, 0.95, 0, 0, 0]", color=GREEN_E)
        self.wait(2)
        self.play(Transform(one_hot, output, replace_mobject_with_target_in_scene=True))
        self.wait(3)
        self.play(FadeOut(output))
        # learning_rate = Text("Learning rate = 0.0001", color=GRAY_E).scale(0.55)
        # learning_rate.next_to(layer_4[0].get_edge_center(DOWN), DOWN)
        # self.play(FadeIn(learning_rate))
        # self.play(*[FadeOut(mob) for mob in self.mobjects])


class NeuralNetworkTransition(MovingCameraScene):
    def construct(self):
        # Transitions
        layer_1 = create_neuron_layer(9, RED_E, [-5, 0, 0], True)
        layer_2 = create_neuron_layer(7, HIDDEN_COL, [-3, 0, 0], True)
        layer_norm_1 = create_neuron_layer(7, NORM_COL, [-2, 0, 0], True)
        layer_drop_1 = create_neuron_layer(7, DROP_COL, [-1, 0, 0], True)
        layer_drop_1_y = create_neuron_layer(7, DROP_COL, [-1, 0, 0], True, dropout=0.5)
        layer_3 = create_neuron_layer(7, HIDDEN_COL, [1, 0, 0], True)
        layer_norm_2 = create_neuron_layer(7, NORM_COL, [2, 0, 0], True)
        layer_drop_2 = create_neuron_layer(7, DROP_COL, [3, 0, 0], True)
        layer_drop_2_y = create_neuron_layer(7, DROP_COL, [3, 0, 0], True, dropout=0.5)
        layer_4 = create_neuron_layer(6, GREEN_E, [5, -0.4, 0], False)

        lines_1 = create_dense(layer_1, layer_2)

        lines_2 = create_dense(layer_2, layer_norm_1, one_to_one=True)
        lines_2_1 = create_dense(layer_norm_1, layer_drop_1)
        lines_2_2 = create_dense(layer_drop_1, layer_3)
        lines_2_1_y = create_dense(layer_norm_1, layer_drop_1_y)
        lines_2_2_y = create_dense(layer_drop_1_y, layer_3)

        lines_3 = create_dense(layer_3, layer_norm_2, one_to_one=True)
        lines_3_1 = create_dense(layer_norm_2, layer_drop_2)
        lines_3_2 = create_dense(layer_drop_2, layer_4)
        lines_3_1_y = create_dense(layer_norm_2, layer_drop_2_y)
        lines_3_2_y = create_dense(layer_drop_2_y, layer_4)

        # Input Layer
        l1_vg = VGroup(*layer_1)
        brace_l1 = Brace(l1_vg, color=GRAY_E, direction=LEFT, sharpness=1.2)
        brace_l1_text = Text("Input size: 561", color=GRAY_E).rotate(np.pi / 2).scale(0.75)
        brace_l1_text.next_to(brace_l1.get_edge_center(LEFT), LEFT)

        # Hidden Layer 1
        l2_vg = VGroup(*layer_2)
        brace_l2 = Brace(l2_vg, color=GRAY_E, direction=UP, sharpness=1.2)
        brace_l2_text_1 = Text("108 Neurons", color=GRAY_E).scale(0.55)
        brace_l2_text_1.next_to(brace_l2, UP)
        brace_l2_text_2 = Text("Sigmoid", color=GRAY_E).scale(0.55)
        brace_l2_text_2.next_to(brace_l2_text_1.get_edge_center(UP), UP, buff=0.1)
        brace_l2_text_3 = Text("Dense", color=GRAY_E).scale(0.55)
        brace_l2_text_3.next_to(brace_l2_text_2.get_edge_center(UP), UP, buff=0.1)

        # Batch Norm 1
        bnorm1_vg = VGroup(*layer_norm_1)
        brace_norm1 = Brace(bnorm1_vg, color=GRAY_E, direction=UP, sharpness=1.2)
        brace_norm1_text_1 = Text("Batch Normalization", color=GRAY_E).scale(0.55)
        brace_norm1_text_1.next_to(brace_norm1, UP)

        # Dropout 1
        dl1_vg = VGroup(*layer_drop_1)
        dl1_vg_y = VGroup(*layer_drop_1_y)
        brace_drop1 = Brace(dl1_vg, color=GRAY_E, direction=UP, sharpness=1.2)
        brace_drop1_text_1 = Text("Drop rate=0.5", color=GRAY_E).scale(0.55)
        brace_drop1_text_1.next_to(brace_drop1, UP)
        brace_drop1_text_2 = Text("Dropout", color=GRAY_E).scale(0.55)
        brace_drop1_text_2.next_to(brace_drop1_text_1.get_edge_center(UP), UP, buff=0.1)

        # Hidden Layer 2
        l3_vg = VGroup(*layer_3)
        brace_l3 = Brace(l3_vg, color=GRAY_E, direction=UP, sharpness=1.2)
        brace_l3_text_1 = Text("72 Neurons", color=GRAY_E).scale(0.55)
        brace_l3_text_1.next_to(brace_l3, UP)
        brace_l3_text_2 = Text("Sigmoid", color=GRAY_E).scale(0.55)
        brace_l3_text_2.next_to(brace_l3_text_1.get_edge_center(UP), UP, buff=0.1)
        brace_l3_text_3 = Text("Dense", color=GRAY_E).scale(0.55)
        brace_l3_text_3.next_to(brace_l3_text_2.get_edge_center(UP), UP, buff=0.1)

        # Batch Norm 2
        bnorm2_vg = VGroup(*layer_norm_2)
        brace_norm2 = Brace(bnorm2_vg, color=GRAY_E, direction=UP, sharpness=1.2)
        brace_norm2_text_1 = Text("Batch Normalization", color=GRAY_E).scale(0.55)
        brace_norm2_text_1.next_to(brace_norm2, UP)

        # Dropout 2
        dl2_vg = VGroup(*layer_drop_2)
        dl2_vg_y = VGroup(*layer_drop_2_y)
        brace_drop2 = Brace(dl2_vg, color=GRAY_E, direction=UP, sharpness=1.2)
        brace_drop2_text_1 = Text("Drop rate=0.5", color=GRAY_E).scale(0.55)
        brace_drop2_text_1.next_to(brace_drop2, UP)
        brace_drop2_text_2 = Text("Dropout", color=GRAY_E).scale(0.55)
        brace_drop2_text_2.next_to(brace_drop2_text_1.get_edge_center(UP), UP, buff=0.1)

        # Output Layer
        l4_vg = VGroup(*layer_4)
        brace_l4 = Brace(l4_vg, color=GRAY_E, direction=UP, sharpness=1.2)
        brace_l4_text_1 = Text("6 Neurons", color=GRAY_E).scale(0.55)
        brace_l4_text_1.next_to(brace_l4, UP)
        brace_l4_text_2 = Text("Softmax", color=GRAY_E).scale(0.55)
        brace_l4_text_2.next_to(brace_l4_text_1.get_edge_center(UP), UP, buff=0.1)
        brace_l4_text_3 = Text("Dense", color=GRAY_E).scale(0.55)
        brace_l4_text_3.next_to(brace_l4_text_2.get_edge_center(UP), UP, buff=0.1)

        self.add(lines_1, lines_2, lines_2_1, lines_2_2, lines_3, lines_3_1, lines_3_2,
                 l1_vg, l2_vg, l3_vg, l4_vg,
                 bnorm1_vg, bnorm2_vg,
                 dl1_vg, dl2_vg)
        self.play(FadeIn(brace_l1), FadeIn(brace_l1_text))
        self.wait(9)
        self.play(self.camera.frame.animate.scale(1.25).move_to(ORIGIN),
                  Transform(brace_l1, brace_l2, replace_mobject_with_target_in_scene=True), FadeOut(brace_l1_text),
                  FadeIn(brace_l2_text_1), FadeIn(brace_l2_text_2), FadeIn(brace_l2_text_3))
        self.wait(5)
        self.play(Transform(brace_l2, brace_norm1, replace_mobject_with_target_in_scene=True),
                  FadeIn(brace_norm1_text_1),
                  FadeOut(brace_l2_text_1), FadeOut(brace_l2_text_2), FadeOut(brace_l2_text_3))
        self.wait(3)
        self.play(Transform(brace_norm1, brace_drop1, replace_mobject_with_target_in_scene=True),
                  FadeIn(brace_drop1_text_1), FadeIn(brace_drop1_text_2), FadeOut(brace_norm1_text_1))
        self.wait(2)
        self.play(Transform(dl1_vg, dl1_vg_y, replace_mobject_with_target_in_scene=True),
                  Transform(lines_2_1, lines_2_1_y, replace_mobject_with_target_in_scene=True),
                  Transform(lines_2_2, lines_2_2_y, replace_mobject_with_target_in_scene=True))
        self.wait(2)
        self.play(Transform(brace_drop1, brace_l3, replace_mobject_with_target_in_scene=True),
                  FadeOut(brace_drop1_text_1), FadeOut(brace_drop1_text_2),
                  FadeIn(brace_l3_text_1), FadeIn(brace_l3_text_2), FadeIn(brace_l3_text_3))
        self.wait(2)
        self.play(Transform(brace_l3, brace_norm2, replace_mobject_with_target_in_scene=True),
                  FadeIn(brace_norm2_text_1),
                  FadeOut(brace_l3_text_1), FadeOut(brace_l3_text_2), FadeOut(brace_l3_text_3))
        self.wait(0.75)
        self.play(Transform(brace_norm2, brace_drop2, replace_mobject_with_target_in_scene=True),
                  FadeIn(brace_drop2_text_1), FadeIn(brace_drop2_text_2), FadeOut(brace_norm2_text_1))
        self.wait(0.75)
        self.play(Transform(dl2_vg, dl2_vg_y, replace_mobject_with_target_in_scene=True),
                  Transform(lines_3_1, lines_3_1_y, replace_mobject_with_target_in_scene=True),
                  Transform(lines_3_2, lines_3_2_y, replace_mobject_with_target_in_scene=True))
        self.wait(0.75)
        self.play(Transform(brace_drop2, brace_l4, replace_mobject_with_target_in_scene=True),
                  FadeIn(brace_l4_text_1), FadeIn(brace_l4_text_2), FadeIn(brace_l4_text_3),
                  FadeOut(brace_drop2_text_1), FadeOut(brace_drop2_text_2))
        self.wait(3)
        self.play(*[FadeOut(mob) for mob in self.mobjects])


class NeuralNetworkBackProp(MovingCameraScene):
    def construct(self):
        # Transitions
        layer_1 = create_neuron_layer(5, RED_E, [-3, 0, 0], False)
        layer_2 = create_neuron_layer(4, HIDDEN_COL, [-1, -0.4, 0], False)
        layer_3 = create_neuron_layer(4, HIDDEN_COL, [1, -0.4, 0], False)
        layer_4 = create_neuron_layer(3, GREEN_E, [3, 0, 0], False)

        output_1 = Tex("0.2", color=GREEN_D).scale(0.65)
        output_1.next_to(layer_4[0].get_edge_center(RIGHT), RIGHT)
        output_2 = Tex("0.1", color=GREEN_D).scale(0.65)
        output_2.next_to(layer_4[1].get_edge_center(RIGHT), RIGHT)
        output_3 = Tex("0.7", color=GREEN_D).scale(0.65)
        output_3.next_to(layer_4[2].get_edge_center(RIGHT), RIGHT)

        outputs = VGroup(*[output_1, output_2, output_3])

        loss_func_start = Tex("loss(", color=GREEN_D).scale(0.55)
        loss_func_start.next_to(layer_4[1].get_edge_center(RIGHT), RIGHT, buff=0.1)

        loss_func_1 = Tex("[0.2, 0.1, 0.7]", color=GREEN_D).scale(0.55)
        loss_func_1.next_to(loss_func_start, RIGHT, buff=0.1)

        loss_func_2 = Tex(", [0, 0, 1]))", color=GREEN_D).scale(0.55)
        loss_func_2.next_to(loss_func_1, RIGHT, buff=0.1)

        loss_func_3 = Tex("=", color=GREEN_D).scale(0.55)
        loss_func_3.next_to(VGroup(*[loss_func_start, loss_func_1, loss_func_2]).get_edge_center(DOWN), DOWN)
        loss_func_end = Tex("0.3567", color=GREEN_D).scale(0.55)
        loss_func_end.next_to(loss_func_3.get_edge_center(DOWN), DOWN)

        derivative = MathTex(r"\frac{\partial C}{\partial w_{jk}^l}", color=GRAY_E).scale(1.25)
        derivative.move_to(loss_func_1.get_center())

        lines_1 = create_dense(layer_1, layer_2)
        lines_2 = create_dense(layer_2, layer_3)
        lines_3 = create_dense(layer_3, layer_4)

        # Input Layer
        l1_vg = VGroup(*layer_1)

        # Hidden Layer 1
        l2_vg = VGroup(*layer_2)

        # Hidden Layer 2
        l3_vg = VGroup(*layer_3)

        # Output Layer
        l4_vg = VGroup(*layer_4)

        f_text_1 = Text("Forward Phase", color=GOLD_E).next_to([0, 1.85, 0], UP)
        f_arrow_1 = Arrow(l1_vg.get_edge_center(UP), l2_vg.get_edge_center(UP), color=GOLD_E, buff=0.3)
        f_arrow_2 = Arrow(l2_vg.get_edge_center(UP), l3_vg.get_edge_center(UP), color=GOLD_E, buff=0.3)
        f_arrow_3 = Arrow(l3_vg.get_edge_center(UP), l4_vg.get_edge_center(UP), color=GOLD_E, buff=0.3)

        b_text_1 = Text("Backward Phase", color=BLUE_E).next_to([0, -1.85, 0], DOWN)
        b_arrow_1 = Arrow(l2_vg.get_edge_center(DOWN), l1_vg.get_edge_center(DOWN), color=BLUE_E, buff=0.3)
        b_arrow_2 = Arrow(l3_vg.get_edge_center(DOWN), l2_vg.get_edge_center(DOWN), color=BLUE_E, buff=0.3)
        b_arrow_3 = Arrow(l4_vg.get_edge_center(DOWN), l3_vg.get_edge_center(DOWN), color=BLUE_E, buff=0.3)

        self.add(lines_1, lines_2, lines_3,
                 l1_vg, l2_vg, l3_vg, l4_vg)
        # f_arrow_1, f_arrow_2, f_arrow_3, f_text_1,
        # b_arrow_1, b_arrow_2, b_arrow_3, b_text_1)
        self.wait(2)
        self.play(FadeIn(f_text_1))
        self.wait()
        self.play(FadeIn(f_arrow_1))
        self.wait()
        self.play(FadeIn(f_arrow_2))
        self.wait()
        self.play(FadeIn(f_arrow_3))
        self.wait(2)
        self.play(FadeIn(outputs))
        self.wait()
        self.play(FadeIn(loss_func_start), Transform(outputs, loss_func_1, replace_mobject_with_target_in_scene=True),
                  FadeIn(loss_func_2), FadeIn(loss_func_3), FadeIn(loss_func_end))
        self.wait(2)
        self.play(FadeOut(loss_func_start), FadeOut(loss_func_1), FadeOut(loss_func_2),
                  FadeOut(loss_func_3), FadeOut(loss_func_end), FadeOut(f_text_1),
                  FadeOut(f_arrow_1), FadeOut(f_arrow_2), FadeOut(f_arrow_3))
        self.wait(3)
        self.play(FadeIn(b_text_1))
        self.play(FadeIn(derivative))
        self.wait()
        self.play(FadeOut(derivative))
        self.wait(14)
        self.play(FadeIn(b_arrow_3))
        self.wait()
        self.play(FadeIn(b_arrow_2))
        self.wait()
        self.play(FadeIn(b_arrow_1))
        self.wait(10)
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(2)


class ActEval(MovingCameraScene):
    def construct(self):
        metric_image = ImageMobject("act_metrics.png")
        heat_image = ImageMobject("act_heat.png")
        labels = Text("1: walking\n2: walking_upstairs\n3: walking_downstairs\n4: sitting\n5: standing\n6: laying",
                      color=GRAY_E, line_spacing=1).scale(0.45)
        labels.next_to(heat_image.get_edge_center(LEFT), LEFT, buff=0.3)
        self.play(FadeIn(metric_image))
        self.wait(5)
        self.play(FadeOut(metric_image))
        self.wait()
        self.play(FadeIn(heat_image), FadeIn(labels))
        self.wait(3)
        self.camera.frame.save_state()
        self.play(self.camera.frame.animate.scale(0.35).move_to([0.2, -0.2, 0.0]))
        self.wait(2)
        self.play(self.camera.frame.animate.scale(1).move_to([0.6, -0.6, 0.0]))
        self.wait(8)
        self.play(self.camera.frame.animate.scale(1).move_to([-0.6, 0.6, 0.0]))
        self.wait(8)
        self.play(Restore(self.camera.frame))
        self.wait()


class ActEval_2(Scene):
    def construct(self):
        axes = Axes(
            y_range=[-0.01, 1.01, 0.2],
            x_range=[-0.01, 25.1, 5],
            x_length=9,
            y_length=6,
            x_axis_config={
                "numbers_to_include": np.arange(0, 25.01, 5),
                "numbers_with_elongated_ticks": np.arange(0, 12.01, 2),
            },
            y_axis_config={
                "numbers_to_include": np.arange(-0.01, 1.01, 0.2),
                "numbers_with_elongated_ticks": np.arange(-0.01, 1.01, 0.2),
            },
            tips=False
        )
        axes.set_color(GRAY_E)
        labels = axes.get_axis_labels(x_label="epoch", y_label="loss")
        labels.set_color(GRAY_E)

        loss_vals = [0.9037785530090332, 0.4086264967918396, 0.29261451959609985, 0.23221570253372192,
                     0.19133029878139496, 0.1653478592634201, 0.14426515996456146, 0.13066624104976654,
                     0.116917684674263, 0.10667255520820618, 0.09900473058223724, 0.0934741273522377,
                     0.08693065494298935, 0.08022071421146393, 0.07359958440065384, 0.07147156447172165,
                     0.0671391636133194, 0.06467203795909882, 0.062149081379175186, 0.057943087071180344,
                     0.05567743256688118, 0.05384262651205063, 0.05185995250940323, 0.0490415059030056,
                     0.0478949248790741]

        loss_lines = create_line_plot(
            [axes.coords_to_point(float(inx), float(val)) for inx, val in enumerate(loss_vals)], BLUE_E, strokewidth=3)

        train_acc = [0.7358058094978333, 0.916655421257019, 0.941335141658783, 0.9540121555328369, 0.9657450914382935,
                     0.9708698391914368, 0.9758597612380981, 0.9770734906196594, 0.9776129722595215, 0.9805799126625061,
                     0.9820634126663208, 0.9824679493904114, 0.9846257567405701, 0.9840863347053528, 0.9847606420516968,
                     0.9865138530731201, 0.9874578714370728, 0.986648678779602, 0.9889413118362427, 0.98799729347229,
                     0.9892110824584961, 0.9870532751083374, 0.9878624677658081, 0.9898853898048401, 0.989345908164978]

        train_acc_lines = create_line_plot(
            [axes.coords_to_point(float(inx), float(val)) for inx, val in enumerate(train_acc)], BLUE_E,
            strokewidth=3)

        vali_acc = [0.5734312534332275, 0.885847806930542, 0.9329105615615845, 0.942923903465271, 0.9469292163848877,
                    0.9422563314437866, 0.9502670168876648, 0.9495994448661804, 0.9098798036575317, 0.9439252614974976,
                    0.9419225454330444, 0.9472630023956299, 0.8868491053581238, 0.9282376766204834, 0.9405874609947205,
                    0.9392523169517517, 0.9609479308128357, 0.8808411359786987, 0.9552736878395081, 0.8768357634544373,
                    0.9532710313796997, 0.9315754175186157, 0.9355807900428772, 0.9465954899787903, 0.9319092035293579]
        vali_acc_lines = create_line_plot(
            [axes.coords_to_point(float(inx), float(val)) for inx, val in enumerate(vali_acc)], GOLD_E, strokewidth=3)

        train_text = Text("Training", color=BLUE_E).scale(0.55)
        train_text.move_to(ORIGIN, RIGHT)
        vali_text = Text("Valiadtion", color=GOLD_E).scale(0.55)
        vali_text.next_to(train_text, DOWN)

        self.add(axes)
        self.play(FadeIn(labels), FadeIn(loss_lines))
        self.wait(6)

        labels_1 = axes.get_axis_labels(x_label="epoch", y_label="accuracy")
        labels_1.set_color(GRAY_E)

        self.play(Transform(labels, labels_1), Transform(loss_lines, train_acc_lines), FadeIn(vali_acc_lines),
                  FadeIn(train_text), FadeIn(vali_text))
        self.wait(5.5)


class TransEval(MovingCameraScene):
    def construct(self):
        metric_image = ImageMobject("trans_metrics.png")
        heat_image = ImageMobject("trans_heat.png")
        labels = Text(
            "7: stand_to_sit\n8: sit_to_stand\n9: sit_to_lie\n10: lie_to_sit\n11: stand_to_lie\n12: lie_to_stand",
            color=GRAY_E, line_spacing=1).scale(0.45)
        labels.next_to(heat_image.get_edge_center(LEFT), LEFT, buff=0.3)
        self.play(FadeIn(heat_image), FadeIn(labels))
        self.wait(5)
        self.camera.frame.save_state()
        self.play(self.camera.frame.animate.scale(0.35).move_to([0.2, -0.2, 0.0]))
        self.wait(10)
        self.play(self.camera.frame.animate.scale(1).move_to([-0.5, 0.4, 0.0]))
        self.wait(10)
        self.play(Restore(self.camera.frame), FadeOut(heat_image, labels))
        self.wait(5)
        self.play(FadeIn(metric_image))
        self.wait(5)
        self.play(FadeOut(metric_image))


class TransEval_2(Scene):
    def construct(self):
        axes_1 = Axes(
            y_range=[-0.01, 1.81, 0.25],
            x_range=[-0.01, 25.1, 5],
            x_length=9,
            y_length=6,
            x_axis_config={
                "numbers_to_include": np.arange(0, 25.01, 5),
                "numbers_with_elongated_ticks": np.arange(0, 12.01, 2),
            },
            y_axis_config={
                "numbers_to_include": np.arange(-0.01, 1.81, 0.25),
                "numbers_with_elongated_ticks": np.arange(-0.01, 1.81, 0.25),
            },
            tips=False
        )
        axes_1.set_color(GRAY_E)
        labels_1 = axes_1.get_axis_labels(x_label="epoch", y_label="loss")
        labels_1.set_color(GRAY_E)

        axes_2 = Axes(
            y_range=[-0.01, 1.01, 0.2],
            x_range=[-0.01, 25.1, 5],
            x_length=9,
            y_length=6,
            x_axis_config={
                "numbers_to_include": np.arange(0, 25.01, 5),
                "numbers_with_elongated_ticks": np.arange(0, 12.01, 2),
            },
            y_axis_config={
                "numbers_to_include": np.arange(-0.01, 1.01, 0.2),
                "numbers_with_elongated_ticks": np.arange(-0.01, 1.01, 0.2),
            },
            tips=False
        )
        axes_2.set_color(GRAY_E)
        labels_2 = axes_2.get_axis_labels(x_label="epoch", y_label="accuracy")
        labels_2.set_color(GRAY_E)

        loss_vals = [1.7942084074020386, 1.3930131196975708, 1.3257185220718384, 1.1724426746368408, 1.0197484493255615,
                     0.9329210519790649, 0.8799552917480469, 0.8353474140167236, 0.7719369530677795, 0.7892515063285828,
                     0.7059762477874756, 0.6398428082466125, 0.6638794541358948, 0.5941770076751709, 0.5353838801383972,
                     0.5099930167198181, 0.48762035369873047, 0.4830043911933899, 0.45118972659111023,
                     0.41844990849494934, 0.38263022899627686, 0.3651079833507538, 0.3652034103870392,
                     0.34722068905830383, 0.3489958941936493]

        loss_lines = create_line_plot(
            [axes_1.coords_to_point(float(inx), float(val)) for inx, val in enumerate(loss_vals)], BLUE_E,
            strokewidth=3)

        train_acc = [0.2613636255264282, 0.4318181872367859, 0.46590909361839294, 0.5, 0.6107954382896423,
                     0.6107954382896423, 0.625, 0.6590909361839294, 0.7102271914482117, 0.6988636255264282,
                     0.7301136255264282, 0.7556818127632141, 0.7244318127632141, 0.7670454382896423, 0.8039772510528564,
                     0.8153409361839294, 0.8323863744735718, 0.8238636255264282, 0.8380681872367859, 0.8465909361839294,
                     0.8948863744735718, 0.8863636255264282, 0.8892045617103577, 0.8579545617103577, 0.8863636255264282]

        train_acc_lines = create_line_plot(
            [axes_2.coords_to_point(float(inx), float(val)) for inx, val in enumerate(train_acc)], BLUE_E,
            strokewidth=3)

        vali_acc = [0.3614457845687866, 0.33734938502311707, 0.4397590458393097, 0.4939759075641632, 0.5783132314682007,
                    0.4819277226924896, 0.6024096608161926, 0.6927710771560669, 0.6265060305595398, 0.7228915691375732,
                    0.6566265225410461, 0.6867470145225525, 0.7409638166427612, 0.7289156913757324, 0.7530120611190796,
                    0.7530120611190796, 0.7349397540092468, 0.7650602459907532, 0.7590360641479492, 0.7891566157341003,
                    0.7710843086242676, 0.7530120611190796, 0.7590360641479492, 0.7469879388809204, 0.7771084308624268]
        vali_acc_lines = create_line_plot(
            [axes_2.coords_to_point(float(inx), float(val)) for inx, val in enumerate(vali_acc)], GOLD_E, strokewidth=3)

        train_text = Text("Training", color=BLUE_E).scale(0.55)
        train_text.move_to(ORIGIN, RIGHT)
        vali_text = Text("Valiadtion", color=GOLD_E).scale(0.55)
        vali_text.next_to(train_text, DOWN)

        self.add(axes_2, labels_2)
        self.play(FadeIn(train_acc_lines), FadeIn(vali_acc_lines), FadeIn(train_text), FadeIn(vali_text))
        self.wait(1.5)
        self.play(Transform(axes_2, axes_1), Transform(train_acc_lines, loss_lines), Transform(labels_2, labels_1),
                  FadeOut(vali_acc_lines), FadeOut(train_text), FadeOut(vali_text))
        self.wait(2)
        self.play(*[FadeOut(mob) for mob in self.mobjects])

