ImgDataConvertCodeGen
=====================

Image Data Conversion Code Generation Across Python Libraries for Semantic Interoperability.

Issue

* version of image processing libraries


Usage
-----
.. code-block:: python

    source_image = ...
    # target variable name. This will be used to store the result of the conversion.
    target_var = 'target_result'

    # Conversion code generation
    convert_code = code_generator.generate_code('source_image', source_metadata,
                                                target_var, target_metadata)

    # Image data conversion and store the result in the target variable
    exec(convert_code)


Usage
-----
.. code-block:: python


source_image = ...
# define the target variable name. This will be used to store the result of the conversion.
target_var = 'target_result'

# conversion code generator
convert_code = code_generator.generate_code('source_image', image_data['source_metadata'],
                                            target_var, image_data['target_metadata'])
# image data conversion and store the result in the target variable
exec(convert_code)









API
addnodes,
addedge,
addEdgefactory,
build_knowledge
add preset for libraries.


code generator

image data convert
API two interfaces
