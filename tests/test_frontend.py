"""Tests for the dependency-free browser client."""

import unittest

from fastapi.testclient import TestClient


class FrontendTests(unittest.TestCase):
    def test_ui_serves_the_browser_client(self):
        from api import server

        client = TestClient(server.app)
        response = client.get("/ui")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("text/html"))
        self.assertIn("CareerGraph AI", response.text)
        self.assertIn("/query/auto", response.text)

    def test_root_remains_the_json_health_endpoint(self):
        from api import server

        client = TestClient(server.app)
        response = client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")


if __name__ == "__main__":
    unittest.main()
