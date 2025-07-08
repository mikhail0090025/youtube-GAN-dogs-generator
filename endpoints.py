import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse
from PIL import Image
import io
import gan
import matplotlib.pyplot as plt

app = FastAPI()

@app.get("/generate_image")
async def generate_image():
    gan.generator.eval()
    with torch.no_grad():
        noise = torch.randn((1, gan.noise_size))
        generated = gan.generator(noise)
        # Денормализация из [0, 1] в [0, 255]
        generated = (generated * 255).clamp(0, 255).byte()
        # Преобразование в PIL Image
        img = Image.fromarray(generated[0].permute(1, 2, 0).cpu().numpy())
        # Сохранение в байты
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
    return Response(content=img_byte_arr, media_type="image/png")

# Эндпоинт для запуска обучения
@app.get("/train/{epochs}")
async def train(epochs: int = 1):
    D_loss, G_loss, KL_div = gan.go_epochs(epochs, gan.discriminator, gan.generator, gan.dogs_dataloader, gan.optimizer_D, gan.optimizer_G, gan.noise_size, gan.g_loss, gan.d_loss)
    return {"D_loss": D_loss, "G_loss": G_loss, "KL": KL_div}

# HTML-страница для визуализации
@app.get("/graph")
async def get_loss_plot():
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(gan.all_D_losses) + 1)
    plt.plot(epochs, gan.all_D_losses, 'b-', label='Discriminator Loss')
    plt.plot(epochs, gan.all_G_losses, 'r-', label='Generator Loss')
    plt.plot(epochs, gan.all_KLs, 'g-', label='KL')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('GAN Training Loss')
    plt.legend()
    plt.grid(True)

    img_byte_arr = io.BytesIO()
    plt.savefig(img_byte_arr, format='PNG')
    plt.close()
    img_byte_arr = img_byte_arr.getvalue()
    return Response(content=img_byte_arr, media_type="image/png")

# Обновлённая HTML-страница
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>GAN Dogs Generator</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; margin-top: 50px; }
        img { max-width: 100%; height: auto; }
        button { padding: 10px 20px; font-size: 16px; margin: 5px; }
        .container { display: flex; flex-direction: column; align-items: center; }
        #loss_plot { max-width: 80%; margin-top: 20px; }
    </style>
</head>
<body>
    <h1>Dogs Generator with GAN</h1>
    <div class="container">
        <button onclick="generate()">Generate New Dog</button>
        <br><br>
        <img id="generated_image" src="" alt="Generated Dog">
        <img id="generated_image2" src="" alt="Generated Dog" width="128px">
        <img id="generated_image3" src="" alt="Generated Dog" width="256px">
        <br><br>
        <button onclick="updateLossPlot()">Show Loss Plot</button>
        <img id="loss_plot" src="/graph" alt="Loss Plot">
    </div>
    <script>
        async function generate() {
            const response = await fetch('/generate_image');
            const blob = await response.blob();
            document.getElementById('generated_image').src = URL.createObjectURL(blob);
            document.getElementById('generated_image2').src = URL.createObjectURL(blob);
            document.getElementById('generated_image3').src = URL.createObjectURL(blob);
        }

        async function updateLossPlot() {
            const response = await fetch('/graph');
            const blob = await response.blob();
            // document.getElementById('loss_plot').src = URL.createObjectURL(blob);
        }

        // Генерация при загрузке страницы
        window.onload = generate;
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def get_html():
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)