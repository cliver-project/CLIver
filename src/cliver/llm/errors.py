"""Error handling utilities for LLM connections and responses."""

import logging
from openai import APIConnectionError
from httpx import ConnectError, TimeoutException
import socket
import ssl


logger = logging.getLogger(__name__)


def is_connection_error(error: Exception) -> bool:
    """
    Check if the error is related to network connection issues.
    
    Args:
        error: Exception to check
        
    Returns:
        True if the error is a connection-related error
    """
    connection_error_types = (
        APIConnectionError,
        ConnectError,
        TimeoutException,
        socket.gaierror,  # Address-related errors
        ConnectionRefusedError,
        ConnectionError,
        ssl.SSLError,
        ssl.CertificateError,
    )
    return isinstance(error, connection_error_types)


def get_connection_error_message(error: Exception) -> str:
    """
    Get a user-friendly error message for connection-related errors.
    
    Args:
        error: The exception that occurred
        
    Returns:
        A user-friendly error message
    """
    if is_connection_error(error):
        error_type = type(error).__name__
        original_message = str(error)
        
        # More specific messages based on the error type
        if isinstance(error, APIConnectionError):
            return (
                "Network connection error: Unable to reach the LLM provider. "
                "Please check your internet connection and the provider's service status."
            )
        elif isinstance(error, ConnectError):
            return (
                "Connection error: Failed to connect to the LLM provider. "
                "Please verify your network connection and the API endpoint URL."
            )
        elif isinstance(error, TimeoutException):
            return (
                "Connection timeout: The LLM provider is taking too long to respond. "
                "Please check your network connection or try again later."
            )
        elif isinstance(error, (socket.gaierror, ConnectionRefusedError)):
            return (
                "Network error: Unable to connect to the LLM provider. "
                "Please check if the service is running and accessible."
            )
        elif isinstance(error, ssl.SSLError):
            return (
                "SSL/Security error: Unable to establish a secure connection to the LLM provider. "
                "Please check your SSL certificates and network security settings."
            )
        else:
            # Generic connection error message
            return (
                f"Connection error: {original_message}. "
                "Please check your network connection and the LLM provider's availability."
            )
    
    # If it's not a connection error, return the original error
    return str(error)


def get_friendly_error_message(error: Exception, context: str = "LLM operation") -> str:
    """
    Get a user-friendly error message for various types of errors.
    
    Args:
        error: The exception that occurred
        context: Context of where the error occurred
        
    Returns:
        A user-friendly error message
    """
    if is_connection_error(error):
        return get_connection_error_message(error)
    
    # Handle other specific error types
    error_type = type(error).__name__
    
    if error_type == "AuthenticationError":
        return (
            "Authentication error: Invalid API key or authentication credentials. "
            "Please check your API key configuration."
        )
    elif error_type == "RateLimitError":
        return (
            "Rate limit error: You have exceeded the API rate limit. "
            "Please wait before making more requests or upgrade your plan."
        )
    elif error_type == "BadRequestError":
        return (
            "Bad request error: The request to the LLM provider was malformed. "
            f"This might be due to invalid parameters or model configuration. Error details: {str(error)}"
        )
    elif error_type == "NotFound":
        return (
            "Not found error: The requested resource (model, endpoint, etc.) was not found. "
            "Please check your model name and configuration."
        )
    elif error_type == "ServiceUnavailableError":
        return (
            "Service unavailable: The LLM provider's service is currently unavailable. "
            "Please try again later."
        )
    
    # Generic error message
    return f"{context} failed: {error_type} - {str(error)}. Please check your configuration and connection."