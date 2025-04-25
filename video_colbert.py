# video_colbert.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from PIL import Image
import numpy as np
from decord import VideoReader, cpu


class TemporalEncoder(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_layers=4,
        dim_feedforward=2048,
        dropout=0.1,
        num_visual_expansion_tokens=2
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_visual_expansion_tokens = num_visual_expansion_tokens
        
        # Visual expansion tokens (learnable)
        self.visual_expansion_tokens = nn.Parameter(
            torch.zeros(1, num_visual_expansion_tokens, d_model)
        )
        nn.init.normal_(self.visual_expansion_tokens, std=0.02)
        
        # Temporal transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, frame_features):
        """
        Input:
            frame_features: Tensor of shape [batch_size, num_frames, d_model]
        
        Returns:
            video_features: Tensor of shape [batch_size, num_frames + num_visual_expansion_tokens, d_model]
        """
        batch_size = frame_features.shape[0]
        
        # Expand visual expansion tokens to batch size
        expansion_tokens = self.visual_expansion_tokens.expand(batch_size, -1, -1)
        
        # Concatenate frame features with visual expansion tokens
        combined_features = torch.cat([frame_features, expansion_tokens], dim=1)
        
        # Pass through transformer
        video_features = self.transformer(combined_features)
        
        return video_features


class VideoColBERT(nn.Module):
    def __init__(
        self, 
        backbone_name="ViT-B/32", 
        temporal_layers=4,
        temporal_heads=8,
        temporal_ff_dim=2048,
        temporal_dropout=0.1,
        use_query_expansion=True,
        query_pad_length=32,
        visual_expansion_tokens=2,
        temperature=4.77,
        bias=-12.93
    ):
        super().__init__()
        
        self.backbone_name = backbone_name
        self.use_query_expansion = use_query_expansion
        self.query_pad_length = query_pad_length
        
        # Load CLIP model and get feature dimension
        self.clip_model, _ = clip.load(backbone_name, device="cpu", jit=False)
        self.feature_dim = self.clip_model.visual.output_dim
        
        # Initialize temporal encoder
        self.temporal_encoder = TemporalEncoder(
            d_model=self.feature_dim,
            nhead=temporal_heads,
            num_layers=temporal_layers,
            dim_feedforward=temporal_ff_dim,
            dropout=temporal_dropout,
            num_visual_expansion_tokens=visual_expansion_tokens
        )
        
        # For normalizing features
        self.temperature = nn.Parameter(torch.tensor([temperature]))
        self.bias = nn.Parameter(torch.tensor([bias]))
    
    def encode_query(self, text):
        """
        Encode text queries using CLIP
        
        Args:
            text: Text query or list of queries
            
        Returns:
            query_features: Tensor of shape [batch_size, query_length, d_model]
        """
        # Process input to ensure it's a list
        if isinstance(text, str):
            text = [text]
        
        # Get device
        device = next(self.parameters()).device
        
        # Use CLIP's text encoding directly
        with torch.no_grad():
            text_inputs = clip.tokenize(text, truncate=True).to(device)
            text_features = self.clip_model.encode_text(text_inputs)
            
        # Reshape text features for MMS: [batch_size, 1, d_model]
        text_features = text_features.unsqueeze(1)
        
        # If we need to use query expansion, repeat the features to the desired length
        if self.use_query_expansion:
            batch_size = text_features.shape[0]
            expanded_features = text_features.expand(batch_size, self.query_pad_length, -1)
            # Normalize features
            query_features = F.normalize(expanded_features, dim=-1)
        else:
            # Just use the original features
            query_features = F.normalize(text_features, dim=-1)
        
        return query_features
    
    def encode_video(self, frames):
        """
        Encode video frames to get both frame and video features
        
        Args:
            frames: Tensor of shape [batch_size, num_frames, 3, H, W]
            
        Returns:
            frame_features: Tensor of shape [batch_size, num_frames, d_model]
            video_features: Tensor of shape [batch_size, num_frames + num_expansion, d_model]
        """
        batch_size, num_frames = frames.shape[0], frames.shape[1]
        
        # Reshape to process all frames at once
        frames_flat = frames.view(-1, frames.shape[2], frames.shape[3], frames.shape[4])
        
        # Use CLIP to encode frames
        with torch.no_grad():
            frame_features_flat = self.clip_model.encode_image(frames_flat)
            
        # Reshape back to [batch_size, num_frames, d_model]
        frame_features = frame_features_flat.view(batch_size, num_frames, -1)
        
        # Encode with temporal encoder to get video features
        video_features = self.temporal_encoder(frame_features)
        
        # Normalize features
        frame_features = F.normalize(frame_features, dim=-1)
        video_features = F.normalize(video_features, dim=-1)
        
        return frame_features, video_features
    
    def mean_max_sim(self, query_features, visual_features):
        """
        Compute Mean-Max-Similarity between query tokens and visual features
        
        Args:
            query_features: [batch_size, query_length, d_model]
            visual_features: [batch_size, visual_length, d_model]
            
        Returns:
            similarity: [batch_size]
        """
        # Compute pairwise similarities between all query and visual tokens
        # Shape: [batch_size, query_length, visual_length]
        sim = torch.bmm(query_features, visual_features.transpose(1, 2))
        
        # For each query token, find the maximum similarity across all visual tokens
        # Shape: [batch_size, query_length]
        max_sim = torch.max(sim, dim=2)[0]
        
        # Take the mean across query tokens to get the final similarity score
        # Shape: [batch_size]
        mean_max_sim = torch.mean(max_sim, dim=1)
        
        return mean_max_sim
    
    def compute_similarity(self, query_features, frame_features, video_features):
        """
        Compute the final similarity score using MMS_FV
        
        Args:
            query_features: [batch_size, query_length, d_model]
            frame_features: [batch_size, num_frames, d_model]
            video_features: [batch_size, num_frames + num_expansion, d_model]
            
        Returns:
            similarity: [batch_size]
        """
        # Frame-level MMS
        mms_f = self.mean_max_sim(query_features, frame_features)
        
        # Video-level MMS
        mms_v = self.mean_max_sim(query_features, video_features)
        
        # Final similarity is the sum of frame and video MMS
        similarity = mms_f + mms_v
        
        return similarity, mms_f, mms_v
    
    def forward(self, text, frames):
        """
        Forward pass for the model
        
        Args:
            text: List of text queries
            frames: Tensor of shape [batch_size, num_frames, 3, H, W]
            
        Returns:
            similarities: Similarity matrix of shape [batch_size, batch_size]
            frame_similarities: Frame-level similarity matrix
            video_similarities: Video-level similarity matrix
        """
        # Encode queries and videos
        query_features = self.encode_query(text)
        frame_features, video_features = self.encode_video(frames)
        
        batch_size = query_features.shape[0]
        
        # Initialize similarity matrices
        frame_similarities = torch.zeros(batch_size, batch_size, device=query_features.device)
        video_similarities = torch.zeros(batch_size, batch_size, device=query_features.device)
        
        # Compute similarity between all pairs of queries and videos
        for i in range(batch_size):
            # Get i-th query features (need to repeat for each video)
            q_features = query_features[i:i+1].expand(batch_size, -1, -1)
            
            # Compute similarity between i-th query and all videos
            _, f_sim, v_sim = self.compute_similarity(q_features, frame_features, video_features)
            
            # Store in similarity matrices
            frame_similarities[i] = f_sim
            video_similarities[i] = v_sim
        
        # Final similarity is the sum of frame and video similarities
        similarities = frame_similarities + video_similarities
        
        return similarities, frame_similarities, video_similarities
    
    def get_video_representations(self, frames):
        """
        Get video representations for indexing
        
        Args:
            frames: Tensor of shape [num_frames, 3, H, W]
            
        Returns:
            frame_features: Tensor of shape [num_frames, d_model]
            video_features: Tensor of shape [num_frames + num_expansion, d_model]
        """
        # Add batch dimension
        frames = frames.unsqueeze(0)
        
        # Get features
        frame_features, video_features = self.encode_video(frames)
        
        # Remove batch dimension
        frame_features = frame_features.squeeze(0)
        video_features = video_features.squeeze(0)
        
        return frame_features, video_features
    
    def get_query_representation(self, text):
        """
        Get query representation for retrieval
        
        Args:
            text: Text query
            
        Returns:
            query_features: Tensor of shape [query_length, d_model]
        """
        # Get features
        query_features = self.encode_query(text)
        
        # Remove batch dimension if single query
        if isinstance(text, str):
            query_features = query_features.squeeze(0)
            
        return query_features
    
    def retrieve(self, query_features, frame_features_list, video_features_list):
        """
        Retrieve videos based on query features
        
        Args:
            query_features: Tensor of shape [query_length, d_model]
            frame_features_list: List of frame features for each video
            video_features_list: List of video features for each video
            
        Returns:
            scores: List of similarity scores
        """
        # Add batch dimension to query features
        query_features = query_features.unsqueeze(0)
        
        scores = []
        
        # Compute similarity with each video
        for frame_feat, video_feat in zip(frame_features_list, video_features_list):
            # Add batch dimension
            frame_feat = frame_feat.unsqueeze(0)
            video_feat = video_feat.unsqueeze(0)
            
            # Compute similarity
            sim, _, _ = self.compute_similarity(query_features, frame_feat, video_feat)
            scores.append(sim.item())
            
        return scores


def load_video_frames(video_path, num_frames=12, frame_size=224):
    """
    Load frames from a video file
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to sample
        frame_size: Size to resize frames to
        
    Returns:
        frames: Tensor of shape [num_frames, 3, H, W]
    """
    # Read video frames
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    
    # Sample frames uniformly
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = vr.get_batch(indices).asnumpy()
    
    # Process frames
    frames_tensor = []
    for frame in frames:
        # Convert to PIL Image and resize
        img = Image.fromarray(frame).resize((frame_size, frame_size))
        
        # Convert to tensor and normalize
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        
        # Normalize using CLIP mean and std
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        frames_tensor.append(img_tensor)
    
    frames_tensor = torch.stack(frames_tensor)
    
    return frames_tensor