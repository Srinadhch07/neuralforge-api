upload_dataset = """
Upload a leaf image and get predicted plant species and disease with confidence score.


    #API USAGE:

    Input instruction:

    1. Upload zip file only with name of plant & disease in extact follwing naming convention

        ex: mango_rust.zip

        explanation: left part of filename will indicate the plant name and right part will indicate the disease.

    2. The uploading ZIP file must contain the following folder structure:
        
            filename/
                train/
                test/

        example:

            mango_rust/
                train/
                test/

"""