import numpy as np
import cv2
import Task1A_E
import tensorflow as tf
import math
import matplotlib.pyplot as plt

'''
Arguments - Sudoku grid as list of list
Return - None
Description - This function prints the grid in the console
'''
def board_printer(board):
    print("\n\n RECREATED SUDOKU BOARD\n")
    for y in range(len(board)):
        if y % 3 == 0 and y >= 0:
            print("- " * (len(board) + (math.ceil(len(board) / 3))+1))
        for x in range(len(board)):
            if x == len(board) - 1:
                end = "\n"
            elif (x + 1) % 3 == 0:
                end = "| "
            else:
                end = ""
            if x == 0 and board[y][x]:
                print('| {} '.format(board[y][x]), end=end)
            elif x == 0 and not board[y][x]:
                print('|   ', end=end)
            elif x == 8 and board[y][x]:
                print('{} |'.format(board[y][x]), end=end)
            elif x == 8 and not board[y][x]:
                print('  |', end=end)
            else:
                print('{} '.format(board[y][x]) if board[y][x] else "  ", end=end)
    print("- " * (len(board) + (math.ceil(len(board) / 3))+1))
    return

'''
Arguments - None
Return - None
Description - This function starts execution and implements the extension
'''
def main():
    # Read the sudoku image and preprocess
    img = cv2.imread('sudoku.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    array = np.array(img)
    plt.imshow(array)
    array = 255-array
    
    # Divide the sudoku grid into blocks to get number in each block and store it
    divisor = array.shape[0]//9
    puzzle = []
    for i in range(9):
        row = []
        for j in range(9):
            row.append(cv2.resize(array[i*divisor:(i+1)*divisor,
                                        j*divisor:(j+1)*divisor][3:-3, 3:-3], 
                                  dsize=(28,28), 
                                  interpolation=cv2.INTER_CUBIC))
        puzzle.append(row)
    
    # Get the mean of an empty block and a filled block to guess the block which has data
    print(np.mean(puzzle[0][0]))
    print(np.mean(puzzle[0][1]))
    
    # Load model and weights obtained by training MNSIT
    model = Task1A_E.MyModel()
    model.load_weights('DNNweights').expect_partial()
    
    # Store the numbers of grid in an image list and preprocess the image before storing
    count = 0
    grid = [[0 for i in range(9)] for j in range(9)]
    i=0
    j=0
    test_files = []
    indices = []
    for row in puzzle:
        j=0
        for spot in row:
            if np.mean(spot) > 14:
                indices.append((i,j))
                count += 1
                img = cv2.resize(spot, (28, 28))
                img = np.array(img).astype("float32") / 255
                test_files.append(img)
            j+=1
        i+=1
    test_files = np.array(test_files)
    test_files = np.expand_dims(test_files, -1)
    test_files_labels = list(range(10))
    test_files_labels = tf.keras.utils.to_categorical(test_files_labels, 10)
    
    # Predict the digit in image
    predictions = model.predict(test_files)
    for i in range(count):
        grid[indices[i][0]][indices[i][1]] = int(f'{tf.argmax(predictions[i])}')
        
    # Print the board
    board_printer(grid)
    return
    
if __name__ == "__main__":
   main()

