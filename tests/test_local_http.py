import socket
import unittest
from http.server import BaseHTTPRequestHandler
from unittest.mock import patch

from local_http import LocalHTTPServer


class LocalHTTPTest(unittest.TestCase):
    def test_startup_avoids_dns_and_rejects_an_occupied_port(self):
        with patch('socket.getfqdn',side_effect=AssertionError('No reverse DNS')):
            server=LocalHTTPServer(('127.0.0.1',0),BaseHTTPRequestHandler)
            try:
                self.assertEqual(server.server_name,'localhost')
                self.assertGreater(server.server_port,0)
                with self.assertRaises(OSError):
                    duplicate=LocalHTTPServer(server.server_address,BaseHTTPRequestHandler)
                    duplicate.server_close()
            finally:server.server_close()
