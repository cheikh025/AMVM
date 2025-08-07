import os

# Set 'http_proxy' and 'https_proxy' if they are not set
if 'http_proxy' not in os.environ:
    os.environ['http_proxy'] = 'http://127.0.0.1:3128'
    print("http_proxy set to http://127.0.0.1:3128")

if 'https_proxy' not in os.environ:
    os.environ['https_proxy'] = 'http://127.0.0.1:3128'
    print("https_proxy set to http://127.0.0.1:3128")

import os
import ssl
from urllib3.connection import HTTPSConnection
import http.client

def create_unverified_ssl_context():
    """Create and return an SSL context with disabled verification."""
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    return ssl_context

# Ensure that we only patch HTTPSConnection.__init__ once
if not hasattr(HTTPSConnection, 'is_patched'):
    # Backup the original __init__ method of HTTPSConnection
    original_urllib3_init = HTTPSConnection.__init__

    def patched_urllib3_init(self, *args, **kwargs):
        if 'ssl_context' not in kwargs or kwargs['ssl_context'] is None:
            kwargs['ssl_context'] = create_unverified_ssl_context()
        original_urllib3_init(self, *args, **kwargs)

    # Apply the patch
    HTTPSConnection.__init__ = patched_urllib3_init
    # Set a flag to indicate that the patch has been applied
    HTTPSConnection.is_patched = True

# # Patch http.client's HTTPSConnection initialization
# original_http_client_init = http.client.HTTPSConnection.__init__

# def patched_http_client_init(self, *args, **kwargs):
#     original_http_client_init(self, *args, **kwargs)
#     self._context = create_unverified_ssl_context()

# http.client.HTTPSConnection.__init__ = patched_http_client_init

import warnings
import contextlib

import requests
from urllib3.exceptions import InsecureRequestWarning

old_merge_environment_settings = requests.Session.merge_environment_settings

@contextlib.contextmanager
def no_ssl_verification():
    opened_adapters = set()

    def merge_environment_settings(self, url, proxies, stream, verify, cert):
        # Verification happens only once per connection so we need to close
        # all the opened adapters once we're done. Otherwise, the effects of
        # verify=False persist beyond the end of this context manager.
        opened_adapters.add(self.get_adapter(url))

        settings = old_merge_environment_settings(self, url, proxies, stream, verify, cert)
        settings['verify'] = False

        return settings

    requests.Session.merge_environment_settings = merge_environment_settings

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', InsecureRequestWarning)
            yield
    finally:
        requests.Session.merge_environment_settings = old_merge_environment_settings

        for adapter in opened_adapters:
            try:
                adapter.close()
            except:
                pass