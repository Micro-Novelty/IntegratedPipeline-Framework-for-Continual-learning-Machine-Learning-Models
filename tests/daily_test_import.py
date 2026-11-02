def test_smoke():
    import AbstractIntegratedModule
    from AbstractIntegratedModule import IntegratedPipeline
    from AbstractIntegratedModule import Transformer
    from AbstractIntegratedModule import MLP
    assert AbstractIntegratedModule is not None

test_smoke()
