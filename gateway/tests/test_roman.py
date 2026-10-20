from roman import romanize


def test_hindi_is_written_in_latin_letters():
    assert romanize("नमस्ते") == "namaste"
    assert romanize("हिंदी") == "hindi"
    assert romanize("मुझे कल मिलना है") == "mujhe kal milna hai"


def test_english_is_left_alone():
    assert romanize("hello हिंदी") == "hello hindi"
    assert romanize("point 1 home") == "point 1 home"
