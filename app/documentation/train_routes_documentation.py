upload_dataset = """
Upload a dataset ZIP file to train a plant disease model.

## API Usage

### 1. File Naming Convention
Upload a ZIP file using the format:

plant__disease.zip

Example:
- **plant** -> plant name
- **disease** -> disease name
---
### 2. ZIP file structure

The ZIP must contain following folder structure:

    plant__disease/
    train/
    test/

Example:

    mango__rust/
        train/  
        test/
"""