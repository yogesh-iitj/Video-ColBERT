# example.py
import torch
from video_colbert import VideoColBERT, load_video_frames

# Example usage with random input
def main():
    # Initialize model
    model = VideoColBERT(
        backbone_name="ViT-B/32",
        temporal_layers=4,
        temporal_heads=8,
        use_query_expansion=True
    )
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    print(f"Running on device: {device}")
    
    # Generate random video frames (simulate a batch of 2 videos with 12 frames each)
    # In a real scenario, you would load actual video files using load_video_frames()
    batch_size = 2
    num_frames = 12
    frame_size = 224
    random_frames = torch.rand(batch_size, num_frames, 3, frame_size, frame_size).to(device)
    
    # Sample text queries
    queries = [
        "a police officer drives his white car onto a grassy field",
        "a man is playing guitar on stage"
    ]
    
    # Run the model
    print("Computing similarities...")
    with torch.no_grad():
        similarities, frame_similarities, video_similarities = model(queries, random_frames)
    
    # Print results
    print("\nSimilarity Matrix:")
    print(similarities)
    
    print("\nFrame-level Similarity Matrix:")
    print(frame_similarities)
    
    print("\nVideo-level Similarity Matrix:")
    print(video_similarities)
    
    # Example of retrieving from an actual video file
    # Uncomment if you have a video file to test with
    """
    print("\nRetrieving from actual video file...")
    video_path = "sample_video.mp4"
    
    # Load video frames
    frames = load_video_frames(video_path, num_frames=12, frame_size=224).to(device)
    
    # Get query representation
    query = "a police officer drives his white car onto a grassy field"
    query_features = model.get_query_representation(query)
    
    # Get video representations
    frame_features, video_features = model.get_video_representations(frames)
    
    # Compute similarity
    similarity, frame_sim, video_sim = model.compute_similarity(
        query_features.unsqueeze(0),
        frame_features.unsqueeze(0),
        video_features.unsqueeze(0)
    )
    
    print(f"Query: {query}")
    print(f"Similarity score: {similarity.item():.4f}")
    print(f"Frame-level similarity: {frame_sim.item():.4f}")
    print(f"Video-level similarity: {video_sim.item():.4f}")
    """


if __name__ == "__main__":
    main()