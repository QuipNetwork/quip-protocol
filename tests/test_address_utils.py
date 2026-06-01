import pytest
import socket
from shared.address_utils import (
    parse_host_port,
    format_host_port,
    is_ipv6,
    is_ipv4,
    get_socket_family,
    normalize_address,
    is_loopback,
    is_any_address,
    DEFAULT_PORT,
)


class TestParseHostPort:
    """Test suite for parse_host_port function."""

    def test_ipv4_with_port(self):
        """Test parsing IPv4 address with port."""
        host, port = parse_host_port("127.0.0.1:8080")
        assert host == "127.0.0.1"
        assert port == 8080

    def test_ipv4_without_port(self):
        """Test parsing IPv4 address without port uses default."""
        host, port = parse_host_port("127.0.0.1")
        assert host == "127.0.0.1"
        assert port == DEFAULT_PORT

    def test_ipv4_custom_default_port(self):
        """Test custom default port."""
        host, port = parse_host_port("192.168.1.1", default_port=9000)
        assert host == "192.168.1.1"
        assert port == 9000

    def test_ipv6_bracketed_with_port(self):
        """Test parsing bracketed IPv6 address with port."""
        host, port = parse_host_port("[::1]:8080")
        assert host == "::1"
        assert port == 8080

    def test_ipv6_bracketed_without_port(self):
        """Test parsing bracketed IPv6 address without port."""
        host, port = parse_host_port("[::1]")
        assert host == "::1"
        assert port == DEFAULT_PORT

    def test_ipv6_unbracketed(self):
        """Test parsing unbracketed IPv6 address (no port possible)."""
        host, port = parse_host_port("::1")
        assert host == "::1"
        assert port == DEFAULT_PORT

    def test_ipv6_full_address(self):
        """Test parsing full IPv6 address."""
        host, port = parse_host_port("[2001:db8::1]:443")
        assert host == "2001:db8::1"
        assert port == 443

    def test_ipv6_full_unbracketed(self):
        """Test parsing full IPv6 address without brackets."""
        host, port = parse_host_port("2001:db8:85a3::8a2e:370:7334")
        assert host == "2001:db8:85a3::8a2e:370:7334"
        assert port == DEFAULT_PORT

    def test_ipv6_any_address(self):
        """Test parsing IPv6 any address."""
        host, port = parse_host_port("[::]")
        assert host == "::"
        assert port == DEFAULT_PORT

        host, port = parse_host_port("[::]:8080")
        assert host == "::"
        assert port == 8080

    def test_hostname_with_port(self):
        """Test parsing hostname with port."""
        host, port = parse_host_port("example.com:8080")
        assert host == "example.com"
        assert port == 8080

    def test_hostname_without_port(self):
        """Test parsing hostname without port."""
        host, port = parse_host_port("example.com")
        assert host == "example.com"
        assert port == DEFAULT_PORT

    def test_empty_address_raises(self):
        """Test that empty address raises ValueError."""
        with pytest.raises(ValueError, match="Empty address"):
            parse_host_port("")

    def test_invalid_bracket_raises(self):
        """Test that unclosed bracket raises ValueError."""
        with pytest.raises(ValueError, match="missing '\\]'"):
            parse_host_port("[::1")

    def test_invalid_ipv6_in_brackets_raises(self):
        """Test that invalid IPv6 in brackets raises ValueError."""
        with pytest.raises(ValueError, match="Invalid IPv6 address"):
            parse_host_port("[not-an-ipv6]:8080")

    def test_garbage_after_bracket_raises(self):
        """Test that unexpected characters after bracket raise ValueError."""
        with pytest.raises(ValueError, match="Unexpected characters"):
            parse_host_port("[::1]garbage")

    def test_invalid_port_raises(self):
        """Test that invalid port raises ValueError."""
        with pytest.raises(ValueError, match="Invalid port"):
            parse_host_port("[::1]:notaport")

    def test_port_out_of_range_raises(self):
        """Test that port out of range raises ValueError."""
        with pytest.raises(ValueError, match="Port out of range"):
            parse_host_port("127.0.0.1:99999")

    def test_ipv6_mapped_ipv4_with_port(self):
        """Unbracketed ::ffff:x.x.x.x:port normalizes to plain IPv4."""
        host, port = parse_host_port("::ffff:103.188.95.31:20049")
        assert host == "103.188.95.31"
        assert port == 20049

    def test_ipv6_mapped_ipv4_second_addr(self):
        """Another mapped address seen in production logs."""
        host, port = parse_host_port("::ffff:27.59.125.162:20049")
        assert host == "27.59.125.162"
        assert port == 20049

    def test_ipv6_mapped_ipv4_without_port(self):
        """Bare ::ffff:x.x.x.x normalizes to plain IPv4."""
        host, port = parse_host_port("::ffff:103.188.95.31")
        assert host == "103.188.95.31"
        assert port == DEFAULT_PORT

    def test_ipv6_mapped_ipv4_bracketed_with_port(self):
        """Bracketed [::ffff:x.x.x.x]:port normalizes to plain IPv4."""
        host, port = parse_host_port("[::ffff:103.188.95.31]:20049")
        assert host == "103.188.95.31"
        assert port == 20049

    def test_pure_ipv6_with_port_bracketed(self):
        """Pure IPv6 with port must use brackets and is preserved."""
        host, port = parse_host_port("[2001:db8::1]:20049")
        assert host == "2001:db8::1"
        assert port == 20049


class TestFormatHostPort:
    """Test suite for format_host_port function."""

    def test_format_ipv4(self):
        """Test formatting IPv4 address."""
        result = format_host_port("127.0.0.1", 8080)
        assert result == "127.0.0.1:8080"

    def test_format_ipv6(self):
        """Test formatting IPv6 address uses brackets."""
        result = format_host_port("::1", 8080)
        assert result == "[::1]:8080"

    def test_format_ipv6_full(self):
        """Test formatting full IPv6 address."""
        result = format_host_port("2001:db8::1", 443)
        assert result == "[2001:db8::1]:443"

    def test_format_hostname(self):
        """Test formatting hostname."""
        result = format_host_port("example.com", 80)
        assert result == "example.com:80"


class TestIsIpv6:
    """Test suite for is_ipv6 function."""

    def test_ipv6_localhost(self):
        assert is_ipv6("::1") is True

    def test_ipv6_any(self):
        assert is_ipv6("::") is True

    def test_ipv6_full(self):
        assert is_ipv6("2001:db8:85a3::8a2e:370:7334") is True

    def test_ipv6_with_brackets(self):
        assert is_ipv6("[::1]") is True

    def test_ipv4_not_ipv6(self):
        assert is_ipv6("127.0.0.1") is False

    def test_hostname_not_ipv6(self):
        assert is_ipv6("example.com") is False


class TestIsIpv4:
    """Test suite for is_ipv4 function."""

    def test_ipv4_localhost(self):
        assert is_ipv4("127.0.0.1") is True

    def test_ipv4_any(self):
        assert is_ipv4("0.0.0.0") is True

    def test_ipv4_regular(self):
        assert is_ipv4("192.168.1.1") is True

    def test_ipv6_not_ipv4(self):
        assert is_ipv4("::1") is False

    def test_hostname_not_ipv4(self):
        assert is_ipv4("example.com") is False


class TestGetSocketFamily:
    """Test suite for get_socket_family function."""

    def test_ipv6_returns_af_inet6(self):
        assert get_socket_family("::1") == socket.AF_INET6
        assert get_socket_family("2001:db8::1") == socket.AF_INET6

    def test_ipv4_returns_af_inet(self):
        assert get_socket_family("127.0.0.1") == socket.AF_INET
        assert get_socket_family("192.168.1.1") == socket.AF_INET

    def test_hostname_returns_af_inet(self):
        assert get_socket_family("example.com") == socket.AF_INET


class TestNormalizeAddress:
    """Test suite for normalize_address function."""

    def test_ipv6_normalization(self):
        # IPv6 addresses are normalized to compressed form
        result = normalize_address("0000:0000:0000:0000:0000:0000:0000:0001")
        assert result == "::1"

    def test_ipv4_unchanged(self):
        result = normalize_address("127.0.0.1")
        assert result == "127.0.0.1"

    def test_hostname_unchanged(self):
        result = normalize_address("example.com")
        assert result == "example.com"

    def test_removes_brackets(self):
        result = normalize_address("[::1]")
        assert result == "::1"


class TestIsLoopback:
    """Test suite for is_loopback function."""

    def test_ipv4_loopback(self):
        assert is_loopback("127.0.0.1") is True
        assert is_loopback("127.0.0.2") is True  # All 127.x.x.x are loopback

    def test_ipv6_loopback(self):
        assert is_loopback("::1") is True

    def test_localhost(self):
        assert is_loopback("localhost") is True

    def test_not_loopback(self):
        assert is_loopback("192.168.1.1") is False
        assert is_loopback("8.8.8.8") is False
        assert is_loopback("2001:db8::1") is False


class TestIsAnyAddress:
    """Test suite for is_any_address function."""

    def test_ipv4_any(self):
        assert is_any_address("0.0.0.0") is True

    def test_ipv6_any(self):
        assert is_any_address("::") is True
        assert is_any_address("[::]") is True

    def test_ipv6_any_expanded(self):
        assert is_any_address("0:0:0:0:0:0:0:0") is True

    def test_not_any_address(self):
        assert is_any_address("127.0.0.1") is False
        assert is_any_address("::1") is False
