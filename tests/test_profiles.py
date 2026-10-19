def test_save_load_roundtrip(tmp_path):
    from dvrrestorer.profiles import save_profile, load_profile
    from dvrrestorer.core.pipeline import Keyframe
    kfs = [Keyframe(0.0, 0.1, 0.2, 0.3), Keyframe(2.5, 0.5, 0.6, 0.7)]
    p = tmp_path / "my.json"
    save_profile(p, "test", kfs, direction_deg=5.0)
    name, loaded, dir_deg = load_profile(p)
    assert name == "test"
    assert loaded == kfs
    assert dir_deg == 5.0


def test_list_profiles_sorted(tmp_path):
    from dvrrestorer.profiles import save_profile, list_profiles
    from dvrrestorer.core.pipeline import Keyframe
    for n in ("beta", "alpha", "gamma"):
        save_profile(tmp_path / f"{n}.json", n, [Keyframe(0, 0, 0, 0)])
    assert list_profiles(tmp_path) == ["alpha", "beta", "gamma"]
