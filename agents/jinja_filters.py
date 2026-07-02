"""
agents/jinja_filters.py

Custom Jinja2 filters that replace the Ansible-only `ipaddr` filter,
using Python's stdlib `ipaddress` module (no extra dependency).

Import register_filters(env) and call it on every jinja_env you
create (in config_agent.py and validation_agent.py) so templates
can use | netmask, | network_addr, and | wildcard.
"""

import ipaddress


def netmask(cidr: str) -> str:
    """'192.168.30.0/24' -> '255.255.255.0'"""
    network = ipaddress.ip_network(cidr, strict=False)
    return str(network.netmask)


def network_addr(cidr: str) -> str:
    """'192.168.30.0/24' -> '192.168.30.0'"""
    network = ipaddress.ip_network(cidr, strict=False)
    return str(network.network_address)


def wildcard(cidr: str) -> str:
    """'10.0.0.0/8' -> '0.255.255.255'"""
    network = ipaddress.ip_network(cidr, strict=False)
    return str(network.hostmask)


def register_filters(env):
    env.filters["netmask"] = netmask
    env.filters["network_addr"] = network_addr
    env.filters["wildcard"] = wildcard