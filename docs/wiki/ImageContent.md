# ImageContent

The `Image` class handles image data for multimodal AI interactions.

## Import

```python
from pyai.multimodal import Image
```

## Creating Images

### From File

```python
image = Image.from_file("photo.jpg")
```

### From URL

```python
image = Image.from_url("https://example.com/image.png")
```

### From Bytes

```python
with open("image.png", "rb") as f:
    image = Image.from_bytes(f.read(), media_type="image/png")
```

### From Base64

```python
image = Image.from_base64(
    base64_string,
    media_type="image/jpeg"
)
```

### From PIL Image

```python
from PIL import Image as PILImage

pil_img = PILImage.open("photo.jpg")
image = Image.from_pil(pil_img)
```

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `width` | int | Image width in pixels |
| `height` | int | Image height in pixels |
| `media_type` | str | MIME type (image/jpeg, etc.) |
| `size_bytes` | int | File size in bytes |
| `format` | str | Image format (png, jpeg, etc.) |

## Methods

### resize()

Resize image while maintaining aspect ratio:

```python
# Resize to max dimensions
resized = image.resize(max_width=1024, max_height=1024)

# Resize to specific size
resized = image.resize(width=800, height=600)

# Scale by percentage
resized = image.resize(scale=0.5)  # 50% size
```

### convert()

Convert to different format:

```python
# Convert to JPEG
jpeg_image = image.convert(format="jpeg", quality=85)

# Convert to PNG
png_image = image.convert(format="png")

# Convert to WebP
webp_image = image.convert(format="webp", quality=80)
```

### crop()

Crop image region:

```python
cropped = image.crop(
    left=100,
    top=50,
    width=400,
    height=300
)
```

### to_base64()

Get base64 encoded string:

```python
b64_string = image.to_base64()
```

### save()

Save to file:

```python
image.save("output.png")
image.save("output.jpg", format="jpeg", quality=90)
```

## Using with Agents

### Single Image

```python
from pyai import ask
from pyai.multimodal import Image

image = Image.from_file("chart.png")
response = ask("Explain this chart", images=[image])
```

### Multiple Images

```python
images = [
    Image.from_file("img1.jpg"),
    Image.from_file("img2.jpg"),
    Image.from_file("img3.jpg")
]

response = ask(
    "What do these images have in common?",
    images=images
)
```

### With Agent

```python
from pyai import Agent
from pyai.multimodal import Image

agent = Agent(
    name="Analyst",
    model="gpt-4o"  # Vision model
)

image = Image.from_url("https://example.com/data.png")
result = agent.run("Analyze this data visualization", images=[image])
```

## Format Support

| Format | Read | Write | Notes |
|--------|------|-------|-------|
| JPEG | ✅ | ✅ | Most efficient for photos |
| PNG | ✅ | ✅ | Best for graphics/screenshots |
| GIF | ✅ | ✅ | Animated GIFs supported |
| WebP | ✅ | ✅ | Good compression |
| BMP | ✅ | ✅ | Uncompressed |
| TIFF | ✅ | ✅ | High quality |

## Provider Formats

Different providers accept different formats:

```python
# OpenAI format
openai_content = image.to_openai_format()

# Anthropic format
anthropic_content = image.to_anthropic_format()

# Auto-detect (used internally)
provider_content = image.to_provider_format(provider="openai")
```

## See Also

- [Multimodal-Module](Multimodal-Module) - Module overview
- [AudioContent](AudioContent) - Audio handling
- [VideoContent](VideoContent) - Video handling
