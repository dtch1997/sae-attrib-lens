def test_version():
    # Check that we can import the version number
    from sae_attrib_lens import __version__

    assert isinstance(__version__, str)
    assert __version__ != ""
