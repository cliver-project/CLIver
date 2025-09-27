import pytest
from unittest.mock import patch
from cliver.util import retry_with_confirmation


def failing_function():
    raise Exception("Always fails")


def sometimes_failing_function(should_fail=True):
    if should_fail:
        raise Exception("Failed as expected")
    return "Success"


def test_retry_with_confirmation_success():
    """Test that retry_with_confirmation works when function eventually succeeds."""
    call_count = 0

    def fail_once():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise Exception("First call fails")
        return "Success on second call"

    # Mock input to automatically confirm retries
    with patch('builtins.input', return_value='y'):
        result = retry_with_confirmation(fail_once, max_retries=3)
        assert result == "Success on second call"
        assert call_count == 2


def test_retry_with_confirmation_exhausted_retries():
    """Test that retry_with_confirmation raises exception when retries are exhausted."""
    with patch('builtins.input', return_value='y'):
        with pytest.raises(Exception, match="Always fails"):
            retry_with_confirmation(failing_function, max_retries=2)


def test_retry_with_confirmation_user_declines():
    """Test that retry_with_confirmation stops when user declines to retry."""
    with patch('builtins.input', return_value='n'):
        with pytest.raises(Exception, match="Failed as expected"):
            retry_with_confirmation(sometimes_failing_function, True, max_retries=3)


def test_retry_with_confirmation_no_confirmation():
    """Test that retry_with_confirmation works without user confirmation."""
    call_count = 0

    def fail_once():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise Exception("First call fails")
        return "Success on second call"

    result = retry_with_confirmation(fail_once, max_retries=3, confirm_on_retry=False)
    assert result == "Success on second call"
    assert call_count == 2


def test_retry_with_confirmation_function_args():
    """Test that retry_with_confirmation passes arguments correctly."""
    def function_with_args(a, b, c=None):
        if a == 1 and b == 2 and c == 3:
            return "Correct arguments"
        raise Exception("Wrong arguments")

    result = retry_with_confirmation(
        function_with_args, 1, 2, c=3, max_retries=1, confirm_on_retry=False
    )
    assert result == "Correct arguments"