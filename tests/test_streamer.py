import torch
import pytest
from ai_embedded_dynamic_diversity.models import ModelCore
from ai_embedded_dynamic_diversity.deploy.streamer import AdaptiveStreamer

def test_adaptive_streamer_chunking():
    model = ModelCore(10, 4, 32, 10, 16, 64, 4)
    streamer = AdaptiveStreamer(model, chunk_size=2)
    
    # Large rollout data (8 steps)
    signals = torch.randn(8, 10)
    
    # Process in chunks of 2
    outputs = list(streamer.stream_rollout(signals))
    
    assert len(outputs) == 4 # 8 / 2 = 4 chunks
    assert outputs[0].shape == (2, 4)
    assert outputs[-1].shape == (2, 4)

def test_adaptive_streamer_memory_persistence():
    model = ModelCore(10, 4, 32, 10, 16, 64, 4)
    streamer = AdaptiveStreamer(model, chunk_size=1)
    
    # Signal that depends on memory
    signals = torch.randn(2, 10)
    
    # Run twice, memory should persist
    out1 = list(streamer.stream_rollout(signals[:1]))[0]
    out2 = list(streamer.stream_rollout(signals[1:]))[0]
    
    # If we didn't preserve memory, running separately vs together would differ
    # (assuming deterministic model components)
    streamer_unified = AdaptiveStreamer(model, chunk_size=2)
    unified_outs = list(streamer_unified.stream_rollout(signals))[0]
    
    assert torch.allclose(out1, unified_outs[0:1])
    assert torch.allclose(out2, unified_outs[1:2])
