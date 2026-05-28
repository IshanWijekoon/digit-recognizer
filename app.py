import pygame
import numpy as np
import tensorflow as tf

# 1. Load the model
model = tf.keras.models.load_model('mnist_model.keras')

pygame.init()
WIDTH, HEIGHT = 280, 320 # Increased height to give text its own safe space
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Digit Recognizer MVP")
font = pygame.font.Font(None, 36)

# Colors
BLACK, WHITE, GREEN = (0, 0, 0), (255, 255, 255), (0, 255, 0)
BRUSH_SIZE = 12

# 2. SEPARATE UI FROM DATA: Create a dedicated drawing canvas
canvas = pygame.Surface((280, 280))
canvas.fill(BLACK)

def predict_digit(surface):
    # Capture ONLY the dedicated drawing canvas
    img = pygame.surfarray.array3d(surface)
    img = np.mean(img, axis=2)
    img = np.rot90(img, -1)
    img = np.fliplr(img)
    
    # 3. BOUNDING BOX & CENTERING LOGIC (The Secret Sauce)
    coords = np.argwhere(img > 0) # Find all pixels you drew
    if len(coords) > 0:
        # Get the edges of your drawing
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        # Crop the image tight around the drawing
        cropped = img[y_min:y_max+1, x_min:x_max+1]
        
        # Make the cropped area a perfect square to prevent stretching
        h, w = cropped.shape
        size = max(h, w)
        square = np.zeros((size, size))
        y_off = (size - h) // 2
        x_off = (size - w) // 2
        square[y_off:y_off+h, x_off:x_off+w] = cropped
        
        # Resize to exactly 20x20, then pad with 4 pixels to make it 28x28
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
            # Only draw if clicking inside the canvas area
            if pygame.mouse.get_pos()[1] < 280:
                drawing = True
            
        if event.type == pygame.MOUSEBUTTONUP:
            if drawing:
                drawing = False
                current_prediction = predict_digit(canvas) # Pass canvas, NOT screen!
            
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:
                canvas.fill(BLACK)
                current_prediction = None

    if drawing:
        mouse_pos = pygame.mouse.get_pos()
        # Draw on the CANVAS, not the screen
        pygame.draw.circle(canvas, WHITE, mouse_pos, BRUSH_SIZE)

    # Render loop
    screen.fill(BLACK)
    screen.blit(canvas, (0, 0)) # Paste the pure canvas onto the top of the screen
    
    # Draw UI text at the bottom, safely isolated from the model's eyes
    if current_prediction is not None:
        text = font.render(f"Prediction: {current_prediction}", True, GREEN)
        screen.blit(text, (10, 285))

    pygame.display.flip()

pygame.quit()