"""Address utilities for IPv4 and IPv6 handling in QuIP network."""

import ipaddress
import socket
from typing import Tuple


DEFAULT_PORT = 20049


def parse_host_port(address: str, default_port: int = DEFAULT_PORT) -> Tuple[str, int]:
    """
    Parse a host:port string supporting both IPv4 and IPv6 addresses.

    IPv6 addresses should use bracket notation when combined with a port:
    - "127.0.0.1:8080" -> ("127.0.0.1", 8080)
    - "127.0.0.1" -> ("127.0.0.1", default_port)
    - "[::1]:8080" -> ("::1", 8080)
    - "[::1]" -> ("::1", default_port)
    - "::1" -> ("::1", default_port)
    - "[2001:db8::1]:8080" -> ("2001:db8::1", 8080)

    Args:
        address: The address string to parse
        default_port: Port to use if none specified

    Returns:
        Tuple of (host, port)

    Raises:
        ValueError: If address format is invalid
    """
    address = address.strip()

    if not address:
        raise ValueError("Empty address")

    # Handle bracketed IPv6 addresses: [addr] or [addr]:port
    if address.startswith('['):
        bracket_end = address.find(']')
        if bracket_end == -1:
            raise ValueError(f"Invalid bracketed address (missing ']'): {address}")

        host = address[1:bracket_end]
        remainder = address[bracket_end + 1:]

        # Validate the IPv6 address
        try:
            ipaddress.IPv6Address(host)
        except ipaddress.AddressValueError as e:
            raise ValueError(f"Invalid IPv6 address in brackets: {host}") from e

        if not remainder:
            return (host, default_port)
        elif remainder.startswith(':'):
            port_str = remainder[1:]
            try:
                port = int(port_str)
                if not (0 < port <= 65535):
                    raise ValueError(f"Port out of range: {port}")
                return (host, port)
            except ValueError:
                raise ValueError(f"Invalid port number: {port_str}")
        else:
            raise ValueError(f"Unexpected characters after bracket: {remainder}")

    # Check if this looks like an IPv6 address (contains multiple colons)
    colon_count = address.count(':')

    if colon_count > 1:
        # This is an IPv6 address without brackets (no port)
        try:
            ipaddress.IPv6Address(address)
            return (address, default_port)
        except ipaddress.AddressValueError as e:
            raise ValueError(f"Invalid IPv6 address: {address}") from e

    elif colon_count == 1:
        # Could be IPv4:port or an incomplete IPv6 (rare but valid like "::1" is handled above)
        host, port_str = address.rsplit(':', 1)

        # Try parsing port - if it fails, might be IPv6
        try:
            port = int(port_str)
        except ValueError:
            # Not a valid port number, treat whole thing as address
            # This handles edge cases but typically IPv6 should have more colons
            return (address, default_port)

        # Port parsed successfully, validate range
        if not (0 < port <= 65535):
            raise ValueError(f"Port out of range: {port}")
        return (host, port)

    else:
        # No colons - just a hostname or IPv4 address
        return (address, default_port)


def format_host_port(host: str, port: int) -> str:
    """
    Format a host and port into a proper address string.

    IPv6 addresses are wrapped in brackets.

    Args:
        host: The host address (IPv4, IPv6, or hostname)
        port: The port number

    Returns:
        Formatted address string
    """
    if is_ipv6(host):
        return f"[{host}]:{port}"
    return f"{host}:{port}"


def is_ipv6(address: str) -> bool:
    """
    Check if an address string is an IPv6 address.

    Args:
        address: The address to check (without port)

    Returns:
        True if the address is a valid IPv6 address
    """
    # Remove brackets if present
    addr = address.strip('[]')

    try:
        ipaddress.IPv6Address(addr)
        return True
    except ipaddress.AddressValueError:
        return False


def is_ipv4(address: str) -> bool:
    """
    Check if an address string is an IPv4 address.

    Args:
        address: The address to check (without port)

    Returns:
        True if the address is a valid IPv4 address
    """
    try:
        ipaddress.IPv4Address(address)
        return True
    except ipaddress.AddressValueError:
        return False


def get_socket_family(address: str) -> socket.AddressFamily:
    """
    Determine the appropriate socket address family for an address.

    Args:
        address: The address to check (without port)

    Returns:
        socket.AF_INET6 for IPv6 addresses, socket.AF_INET otherwise
    """
    if is_ipv6(address):
        return socket.AF_INET6
    return socket.AF_INET


def normalize_address(address: str) -> str:
    """
    Normalize an IP address to its canonical form.

    IPv6 addresses are expanded/compressed to standard form.
    IPv4 addresses are returned as-is.

    Args:
        address: The address to normalize

    Returns:
        Normalized address string
    """
    # Remove brackets if present
    addr = address.strip('[]')

    try:
        # Try IPv6 first
        ip = ipaddress.IPv6Address(addr)
        return str(ip)
    except ipaddress.AddressValueError:
        pass

    try:
        # Try IPv4
        ip = ipaddress.IPv4Address(addr)
        return str(ip)
    except ipaddress.AddressValueError:
        pass

    # Return as-is if not a valid IP (might be a hostname)
    return address


def is_loopback(address: str) -> bool:
    """
    Check if an address is a loopback address.

    Args:
        address: The address to check (without port)

    Returns:
        True if the address is a loopback address (127.0.0.1, ::1, localhost)
    """
    addr = address.strip('[]').lower()

    if addr in ('localhost', '127.0.0.1', '::1'):
        return True

    try:
        ip = ipaddress.ip_address(addr)
        return ip.is_loopback
    except ValueError:
        return addr == 'localhost'


def is_any_address(address: str) -> bool:
    """
    Check if an address is the "any" address (0.0.0.0 or ::).

    Args:
        address: The address to check (without port)

    Returns:
        True if the address is 0.0.0.0 or ::
    """
    addr = address.strip('[]')
    return addr in ('0.0.0.0', '::', '0:0:0:0:0:0:0:0')
