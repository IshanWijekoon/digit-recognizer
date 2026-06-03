import pygame
import numpy as np
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('mnist_model.keras')

pygame.init()
CANVAS_SIZE = 360
UI_BOTTOM_SPACE = 70
BRUSH_SIZE = 12
PAD_X, PAD_Y = 10, 10      # Padding to center the canvas
WIDTH = CANVAS_SIZE + (PAD_X * 2)
HEIGHT = CANVAS_SIZE + (PAD_Y * 2) + UI_BOTTOM_SPACE
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Recognizer MVP")
font = pygame.font.Font(None, 36)

# Theme colors
BLACK = (0, 0, 0)          # MUST KEEP: Data Layer Background
WHITE = (255, 255, 255)    # MUST KEEP: Data Layer Brush
DARK_BLUE = (15, 23, 42)   # UI Layer: App Background
CYAN = (56, 189, 248)      # UI Layer: Prediction Text
BORDER = (51, 65, 85)      # UI Layer: Canvas Border


# The data layer
canvas = pygame.Surface((CANVAS_SIZE, CANVAS_SIZE)) 
canvas.fill(BLACK)

def predict_digit(surface):
    # Capture ONLY the black-and-white canvas
    img = pygame.surfarray.array3d(surface)
    img = np.mean(img, axis=2)
    img = np.rot90(img, -1)
    img = np.fliplr(img)
    
    # Bounding Box & Centering
    coords = np.argwhere(img > 0)
    if len(coords) > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        cropped = img[y_min:y_max+1, x_min:x_max+1]
        
        h, w = cropped.shape
        size = max(h, w)
        square = np.zeros((size, size))
        y_off = (size - h) // 2
        x_off = (size - w) // 2
        square[y_off:y_off+h, x_off:x_off+w] = cropped
        
        resized = tf.image.resize(np.expand_dims(square, axis=-1), [20, 20])
        padded = tf.pad(resized, [[4, 4], [4, 4], [0, 0]])
        img = np.squeeze(padded) / 255.0
    else:
        img = np.zeros((28, 28))

    img = img.reshape(1, 28, 28)
    prediction = model.predict(img, verbose=0)
    return np.argmax(prediction)

running, drawing, current_prediction = True, False, None

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Check if click is inside the canvas boundaries
            mouse_x, mouse_y = pygame.mouse.get_pos()
            if PAD_X <= mouse_x <= PAD_X + CANVAS_SIZE and PAD_Y <= mouse_y <= PAD_Y + CANVAS_SIZE:
                drawing = True
            
        if event.type == pygame.MOUSEBUTTONUP:
            if drawing:
                drawing = False
                current_prediction = predict_digit(canvas)
            
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:
                canvas.fill(BLACK)
                current_prediction = None

    if drawing:
        mouse_x, mouse_y = pygame.mouse.get_pos()
        # Adjust mouse coordinates relative to the canvas position
        canvas_x = mouse_x - PAD_X
        canvas_y = mouse_y - PAD_Y
        pygame.draw.circle(canvas, WHITE, (canvas_x, canvas_y), BRUSH_SIZE)

    # Applying the Theme
    screen.fill(DARK_BLUE)
    
    # Draw a subtle border around where the canvas will go
    pygame.draw.rect(screen, BORDER, (PAD_X - 2, PAD_Y - 2, CANVAS_SIZE + 4, CANVAS_SIZE + 4), 2)
    
    # Paste the pure Black & White canvas onto our Dark Blue screen
    screen.blit(canvas, (PAD_X, PAD_Y)) 
    
    # Render Text
    if current_prediction is not None:
        text = font.render(f"Prediction: {current_prediction}", True, CYAN)
        screen.blit(text, (PAD_X, PAD_Y + CANVAS_SIZE + 15))
        

    pygame.display.flip()

pygame.quit()