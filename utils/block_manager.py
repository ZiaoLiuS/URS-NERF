
class blockManager:
    def __init__(self,
                 total_train_number: int,
                 min_block_size: int):

        self.total_train_number = total_train_number
        self.min_block_size = min_block_size
        self.current_block_number = 0

        self.image_idx = [i for i in range(self.total_train_number)]

        self.history_block = []
    def get_current_block(self):
        slice_index = min(self.min_block_size, len(self.image_idx))
        current_image_index = self.image_idx[:slice_index]

        self.image_idx = list(set(self.image_idx) - set(current_image_index))
        self.image_idx.sort()

        self.history_block += current_image_index
        self.history_block.sort()

        return current_image_index


    def get_history_block(self):
        return self.history_block

    def max_block_size(self):
        return self.total_train_number

    def update_block(self,
                     discard_image_index: list):
        self.image_idx += discard_image_index
        self.image_idx.sort()
        self.history_block =  list(set(self.history_block) - set(discard_image_index))
        self.history_block.sort()

    def is_empty(self):
        return len(self.image_idx) == 0


if __name__ == "main":
    blockManager = blockManager(11, 5)
    print("current block: ", blockManager.get_current_block())
    print("history block: ", blockManager.get_history_block())

    blockManager.update_block([2, 3, 4])
    print("current block: ", blockManager.get_current_block())
    print("history block: ", blockManager.get_history_block())
    print("current block: ", blockManager.get_current_block())
    print("history block: ", blockManager.get_history_block())
    print("current block: ", blockManager.get_current_block())
    print("history block: ", blockManager.get_history_block())


