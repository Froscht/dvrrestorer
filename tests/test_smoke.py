def test_package_imports_and_has_version():
    import dvrrestorer
    assert isinstance(dvrrestorer.__version__, str)
    assert dvrrestorer.__version__
