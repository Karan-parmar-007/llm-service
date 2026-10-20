from points import shape_points


def test_spoken_points_become_a_numbered_list():
    text = (
        "mujhe kal milna hai. \n"
        "point 1 server ghar pe rahega. \n"
        "point 2 keyboard android ke liye hai."
    )
    assert shape_points(text) == (
        "mujhe kal milna hai.\n"
        "1. server ghar pe rahega.\n"
        "2. keyboard android ke liye hai."
    )


def test_plain_text_stays_plain():
    assert shape_points("kal milte hain") == "kal milte hain"
