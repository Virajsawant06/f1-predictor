from data_loading import get_raw_dataset
from dnf_features import create_features
from position_features import create_training_dataset
from dnf_train import train_dnf_model
from position_train import train_position_model

if __name__ == '__main__':
    print("Loading data...")
    df = get_raw_dataset("../data/raw")
    
    print("Training DNF model...")
    df_dnf = create_features(df.copy())
    train_dnf_model(df_dnf)
    
    print("Training position model...")
    df_pos = create_training_dataset(df.copy())
    train_position_model(df_pos)
    
    print("All models retrained and saved!")