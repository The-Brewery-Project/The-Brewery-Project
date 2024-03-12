# -*- coding: utf-8 -*-
'''
python script to merge pdf files
'''

# import library
import pypdf

# create pdf merger object
merger = pypdf.PdfWriter()

# pdf name management
pdf1 = '../docs/index.pdf'
pdf2 = '../docs/introduction.pdf'
pdf3 = '../docs/models_implemented.pdf'
pdf4 = '../docs/conclusion.pdf'

pdfs = [pdf1, pdf2, pdf3, pdf4]

# merge pdfs
for pdf in pdfs:
    merger.append(pdf)

# write the final merged pdf into the docs folder
merger.write('../docs/the-brewery-project.pdf')

# close pdf merger object
merger.close()