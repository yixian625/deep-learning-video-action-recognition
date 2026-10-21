"""
Module: models.py

This module defines the LRCN (Long-term Recurrent Convolutional Network) model for video
classification. The LRCN model combines a 2D CNN backbone (e.g., ResNet) for spatial feature
extraction from individual frames with an LSTM to capture temporal dynamics across frames.
An additional fully-connected layer is used to output the final class predictions.

It also provides an S3D 3D CNN (pretrained on Kinetics-400) and a factory to build either model.

Classes:
    Identity: A helper module that returns the input unchanged. It is used to replace the fully-connected
              layer of the ResNet backbone.
    LRCN: The main model that integrates the CNN backbone, an LSTM, dropout regularization, and a final
          fully-connected layer to produce class logits.

Functions:
    build_s3d: Builds an S3D model with a new classification layer.
    build_model: Builds the model selected by model_type ('lrcn' or 's3d').
"""

import torch.nn as nn
from torchvision import models
from torchvision.models.video import s3d, S3D_Weights

class Identity(nn.Module):
    """
    A placeholder identity operator that is argument-insensitive.
    
    This module is used to replace the fully-connected (fc) layer in the ResNet backbone,
    effectively making the backbone output the raw features before classification.
    
    Example:
        >>> identity = Identity()
        >>> output = identity(input_tensor)
    """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        """
        Forward pass that returns the input as is.
        
        Args:
            x (Tensor): Input tensor.
        
        Returns:
            Tensor: The same tensor x.
        """
        return x

class LRCN(nn.Module):
    """
    LRCN (Long-term Recurrent Convolutional Network) for video classification.
    
    This model uses a ResNet backbone as a 2D CNN to extract spatial features from each video frame.
    An LSTM network is then used to model the temporal dynamics across the sequence of frame features.
    Dropout is applied before the final fully-connected layer that produces class logits.

    Args:
        hidden_size (int): Number of features in the hidden state of the LSTM.
        n_layers (int): Number of recurrent layers in the LSTM.
        dropout_rate (float): Dropout rate applied before the final classification layer.
        n_classes (int): Number of output classes.
        pretrained (bool, optional): If True, uses a ResNet model pretrained on ImageNet. Default is True.
        cnn_model (str, optional): Specifies the ResNet variant to use as the backbone.
                                   Options: 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'.
                                   Default is 'resnet34'.
    
    Raises:
        ValueError: If the specified cnn_model is not supported.
    """
    def __init__(self, hidden_size, n_layers, dropout_rate, n_classes, pretrained=True, cnn_model='resnet34'):
        super(LRCN, self).__init__()

        # Set up the ResNet backbone as a 2D CNN feature extractor.
        if cnn_model == 'resnet18':
            base_cnn = models.resnet18(pretrained=pretrained)
        elif cnn_model == 'resnet34':
            base_cnn = models.resnet34(pretrained=pretrained)
        elif cnn_model == 'resnet50':
            base_cnn = models.resnet50(pretrained=pretrained)
        elif cnn_model == 'resnet101':
            base_cnn = models.resnet101(pretrained=pretrained)
        elif cnn_model == 'resnet152':
            # Note: This example uses resnet34 for resnet152 option as a placeholder.
            base_cnn = models.resnet34(pretrained=pretrained)
        else:
            raise ValueError('The input CNN backbone is not supported, please choose a valid ResNet variant.')

        # Retrieve the number of features output by the CNN's original fully-connected layer.
        num_features = base_cnn.fc.in_features
        
        # Replace the original fc layer with an identity mapping so that raw features are returned.
        base_cnn.fc = Identity()
        self.base_model = base_cnn

        # Define the LSTM to process the sequence of frame features.
        # batch_first=True: input is (batch_size, time_steps, features), so the LSTM steps over time
        # within each video instead of over the videos in a batch.
        self.rnn = nn.LSTM(num_features, hidden_size, n_layers, batch_first=True)
        
        # Define dropout for regularization.
        self.dropout = nn.Dropout(dropout_rate)
        
        # Final fully-connected layer to produce logits for each class.
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        """
        Forward pass for the LRCN model.
        
        The input tensor x is expected to have the shape:
            (batch_size, time_steps, channels, height, width)
        
        The CNN backbone extracts features from all frames at once. The per-video sequences of frame
        features are then passed through the LSTM, which steps over time independently for each video.
        The output from the last time step is then passed through dropout and the final fully-connected
        layer to produce the class logits.

        Args:
            x (Tensor): Input tensor of shape (batch_size, time_steps, channels, height, width).

        Returns:
            Tensor: Output logits for each sample in the batch with shape (batch_size, n_classes).
        """
        bs, ts, c, h, w = x.shape  # batch_size, time_steps, channels, height, width
        
        # Extract features for every frame of every video in one pass, then restore the time axis.
        y = self.base_model(x.reshape(bs * ts, c, h, w))
        y = y.reshape(bs, ts, -1)  # (batch_size, time_steps, features)

        # Run the LSTM over the time steps of each video.
        out, _ = self.rnn(y)
        
        # Apply dropout to the output of the final time step.
        out = self.dropout(out[:, -1])
        
        # Pass the final output through the fully-connected layer to get class logits.
        out = self.fc(out)
        return out


def build_s3d(n_classes, pretrained=True, dropout_rate=0.2):
    """
    Build an S3D model for video classification.

    S3D (separable 3D CNN, an Inception-based successor of I3D) takes clips of shape
    (batch_size, channels, time_steps, height, width). With pretrained=True, it is initialized
    from Kinetics-400 weights and its final 1x1x1 convolution is replaced to output n_classes logits.

    Args:
        n_classes (int): Number of output classes.
        pretrained (bool, optional): If True, loads Kinetics-400 pretrained weights. Default is True.
        dropout_rate (float, optional): Dropout rate before the classification layer. Default is 0.2.

    Returns:
        nn.Module: The S3D model.
    """
    model = s3d(weights=S3D_Weights.KINETICS400_V1 if pretrained else None)
    model.classifier[0] = nn.Dropout(p=dropout_rate)
    model.classifier[1] = nn.Conv3d(model.classifier[1].in_channels, n_classes, kernel_size=1, stride=1, bias=True)
    return model


def build_model(model_type, n_classes, pretrained=True, dropout_rate=0.1,
                rnn_hidden_size=100, rnn_n_layers=1, cnn_backbone='resnet34'):
    """
    Build the video classification model selected by model_type.

    Args:
        model_type (str): 'lrcn' (2D CNN + LSTM) or 's3d' (3D CNN).
        n_classes (int): Number of output classes.
        pretrained (bool, optional): Use pretrained weights (ImageNet for LRCN, Kinetics-400 for S3D). Default is True.
        dropout_rate (float, optional): Dropout rate before the classification layer. Default is 0.1.
        rnn_hidden_size (int, optional): LRCN only: number of neurons in the LSTM hidden layer. Default is 100.
        rnn_n_layers (int, optional): LRCN only: number of LSTM layers. Default is 1.
        cnn_backbone (str, optional): LRCN only: ResNet backbone. Default is 'resnet34'.

    Returns:
        nn.Module: The model.

    Raises:
        ValueError: If model_type is not supported.
    """
    if model_type == 'lrcn':
        return LRCN(hidden_size=rnn_hidden_size, n_layers=rnn_n_layers, dropout_rate=dropout_rate,
                    n_classes=n_classes, pretrained=pretrained, cnn_model=cnn_backbone)
    if model_type == 's3d':
        return build_s3d(n_classes, pretrained=pretrained, dropout_rate=dropout_rate)
    raise ValueError('model_type must be either "lrcn" or "s3d".')
