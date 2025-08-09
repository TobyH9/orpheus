from pathlib import Path
import torch


BASE_DIR = Path(__file__).parent.parent.parent.parent

class ParsingData():

    def __init__(self, file_path= BASE_DIR / "resources/tiny_shakespeare.txt"):
        self.file_path = file_path
        self.data_str, self.characters, self.vocab_size = self._load_data()
        self.char_to_int = {ch: i for i, ch in enumerate(self.characters)} # the mapping from the characters in the dataset to integers
        self.int_to_char = {i: ch for i, ch in enumerate(self.characters)} # the mapping from the integers to the characters in the dataset
        self.data_int = torch.tensor(self.characters_to_integers(self.data_str))
        self.train_data = self.data_int[:0.9*len(self.data_int)]
        self.val_data = self.data_int[0.9*len(self.data_int):]

    def _load_data(self) -> tuple[str, list[str], int]:
        """
        Reads the data in the specified path and loads it, finds the set of characters used,
        and the size of the vocabulary.

        Args:
            None

        Returns:
            text (str): a string containing all the text in the given file
            characters (list[str]): a sorted list of strings containing all the characters used in "text"
            vocab_size (int): the length of "characters".
        """
        with open(self.file_path, "r", encoding="utf-8") as f:
            text = f.read()
        characters = sorted(list(set(text)))  # convert the text to a sorted list
        vocab_size = len(characters)
        return text, characters, vocab_size
    
    def characters_to_integers(self, input_string= str) -> list:
        """
        Converts any given string into a list of integers
        based on a pre-defined mapping which is an attribute of the dataset.

        Args:
            input_string (str): the string of characters which you want to
            find the associated integer for.

        Returns:
            output (list[int]): list of integers which are associated with the 
            characters in the input string.
        """
        return [self.char_to_int[c] for c in input_string]

    def integers_to_characters(self, input_list = list) -> str:
        """
        Converts any given list of integers in [0,vocab_size] (inclusive), into a
        string of characters in the vocabulary, based on a mapping which is an attribute of the 
        dataset.
        
        Args:
            input_list (list): the list of integers which you want to
            find the associated string for.

        Returns:
            output (str): str of characters which are associated with
            the integers in the input_list.
        """
        return "".join([self.int_to_char[c] for c in input_list])


data = ParsingData()

print(data.data_int[0:1000])
#encoded = data.characters_to_integers(data.data[1:5])
#decoded = data.integers_to_characters(encoded)
#print(encoded)
#print(decoded)
