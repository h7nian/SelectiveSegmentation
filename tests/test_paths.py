from selectseg.paths import REPOSITORY_ROOT, repository_path


def test_repository_path_normalizes_legacy_nested_checkout_paths():
    assert repository_path("../outputs/example") == REPOSITORY_ROOT / "outputs/example"
    assert repository_path("configs/example.json") == REPOSITORY_ROOT / "configs/example.json"


def test_repository_path_preserves_absolute_paths(tmp_path):
    assert repository_path(tmp_path) == tmp_path
