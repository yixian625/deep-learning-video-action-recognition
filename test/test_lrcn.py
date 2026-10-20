"""
Regression test for the LRCN LSTM recurrence (README "Fixes" #3).

The original LRCN fed each frame to a non-batch_first LSTM as a (batch, 1, features) tensor, so the
LSTM stepped over the videos in a batch instead of over time, and a video's prediction depended on
the other videos in its batch.
"""

import torch

from models import LRCN


def test_prediction_does_not_depend_on_other_videos_in_batch():
    torch.manual_seed(0)
    model = LRCN(hidden_size=16, n_layers=1, dropout_rate=0.0, n_classes=5,
                 pretrained=False, cnn_model='resnet18').eval()
    a, b = torch.randn(1, 4, 3, 32, 32), torch.randn(1, 4, 3, 32, 32)
    with torch.no_grad():
        alone = model(b)[0]
        after_a = model(torch.cat([a, b]))[1]
        before_a = model(torch.cat([b, a]))[0]
    torch.testing.assert_close(after_a, alone)
    torch.testing.assert_close(before_a, alone)
