def test_intentional_ci_failure():
    assert 1 == 0, "Intentional failure to verify that CI blocks failing tests"
