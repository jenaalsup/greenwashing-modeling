import PyPDF2

pdf=open('sustainability-report.pdf', 'rb')
txt=open('sustainability-report.txt', 'a')

pdfreader=PyPDF2.PdfReader(pdf)
for page_num in range(len(pdfreader.pages)): 
    pageobj=pdfreader.pages[page_num]
    text = pageobj.extract_text()
    txt.writelines(text)

txt.close()