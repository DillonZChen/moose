import pddl
import pytest

from moose.planning.regression import check_regression_support
from tests.fixtures import ALL_DOMAINS, get_domain_file


@pytest.mark.parametrize("domain_name", ALL_DOMAINS)
def test_regression_support(domain_name: str):
    domain = pddl.parse_domain(get_domain_file(domain_name))
    check_regression_support(domain)
