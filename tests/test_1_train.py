import pytest

from .fixtures import CLASSIC_DOMAINS, train_routine


@pytest.mark.parametrize("domain_name", CLASSIC_DOMAINS)
def test_train(domain_name: str):
    try:
        train_routine(domain_name)
    except NotImplementedError as e:
        pytest.skip("Config not implemented")
