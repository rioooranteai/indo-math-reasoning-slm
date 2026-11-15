from pathlib import Path

from core.config import PROJECT_DIR, Settings


def test_relative_model_paths_resolve_from_project_folder() -> None:
    settings = Settings(merged_model_path="model/custom", _env_file=None)
    assert settings.merged_model_path == PROJECT_DIR / "model" / "custom"


def test_absolute_model_paths_are_kept(tmp_path: Path) -> None:
    settings = Settings(merged_model_path=tmp_path, _env_file=None)
    assert settings.merged_model_path == tmp_path


def test_generation_defaults() -> None:
    settings = Settings(_env_file=None)
    assert settings.do_sample is True
    assert (settings.temperature, settings.top_p) == (0.7, 0.9)
    assert settings.max_new_tokens == 1024
    assert "\\boxed{nilai}" in settings.system_prompt
