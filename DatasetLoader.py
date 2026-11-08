from datasets import load_dataset
from IPython.display import display
from hugging_face_authentication import hugging_face_auth

class Datasetloader:
    
    DEFULT_DATASET_ID = "mteb/tweet_sentiment_extraction"
    
    def __init__(self, dataset_id: str = DEFULT_DATASET_ID):
        hugging_face_auth()
        self.dataset_id = dataset_id
        self.dataset = load_dataset(dataset_id, split="train")
        self.current_column = None
        
    def checking_dataset(self):
        return self.dataset.to_pandas()
    
    def printing_dataset(self, min = 0, max = None):
        print(self.checking_dataset()[min:max])
    
    def dataset_info(self):
        columns_name = self.dataset.column_names
        
        information = f"The Dataset has {len(columns_name)} columns and are the following:\n {columns_name}"
        print(information.strip())
        
        return information
    
    def ask_user_columns(self):
        user_choice = input(f"these are our colums: {self.dataset.column_names} please choose one to work with \n")
        
        if user_choice in self.dataset.column_names:
            return user_choice
        else:
            print("you didn't enter a vaild column name we will be adding text instead")
            return 'text'
        
    def __repr__(self) -> str:
        return (
            f"Dataset_loader:\n"
            f"Dataset_id = {self.dataset_id}\n"
            f"Rows = {len(self.dataset)}\n"
            f"Colums = {self.dataset.column_names}\n"
        )
if __name__ == "__main__":
    print("==============================================")
    X = Datasetloader()
    display(X.checking_dataset())
    print(X)