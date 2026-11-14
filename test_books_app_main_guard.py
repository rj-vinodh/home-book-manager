import runpy
from unittest.mock import patch

import gradio


def test_books_app_does_not_launch_on_import():
    with patch.object(gradio.Blocks, "launch", autospec=True) as mock_launch:
        runpy.run_path("books_app.py", run_name="not_main")

    mock_launch.assert_not_called()


def test_books_app_launches_when_run_as_main():
    with patch.object(gradio.Blocks, "launch", autospec=True) as mock_launch:
        runpy.run_path("books_app.py", run_name="__main__")

    mock_launch.assert_called_once()
