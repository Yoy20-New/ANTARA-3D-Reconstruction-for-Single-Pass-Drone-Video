import io

def test_pipeline_config_includes_gaussian_flags():
    from antara.config import PipelineConfig

    cfg = PipelineConfig()
    assert cfg.enable_gaussian_splatting is False
    assert cfg.gs_iterations == 7000
    assert cfg.enable_sugar is False


def test_build_parser_supports_gaussian_flags():
    from run_pipeline import build_parser

    parser = build_parser()
    args = parser.parse_args([
        "--video", "demo.mp4",
        "--gaussian-splatting",
        "--sugar",
        "--gs-iterations", "5000",
    ])

    assert args.gaussian_splatting is True
    assert args.sugar is True
    assert args.gs_iterations == 5000


def test_export_mesh_allows_sugar_mesh_path(tmp_path):
    from antara.deliverables import export_mesh

    ply_path = tmp_path / "cloud.ply"
    obj_path = tmp_path / "mesh.obj"
    sugar_obj_path = tmp_path / "sugar.obj"

    ply_path.write_text("ply\nformat ascii 1.0\n")
    sugar_obj_path.write_text("obj file")

    result = export_mesh(str(ply_path), str(obj_path), sugar_mesh_path=str(sugar_obj_path))
    assert result == str(obj_path)
    assert obj_path.exists()
