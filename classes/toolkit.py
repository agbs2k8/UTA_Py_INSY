"""
This will contain the functions that need to be implemented to allow all user functions
To build:
    - Document classification & matching from normalized & scaled Non-negative Matrix Factorization

Completed:
    mongo_connect() - returns MongoDB collection item
    tokenize(text) - pass it a string of text and it will return the list of tokens
    vectorize_single() - pass it a string of a full text document and it will return the TF-IDF vector as a CSR-Mat
    download_pdf_from_url(url): Pass it URL for a PDF file that needs to be downloaded.
                                Success returns tuple: binary of PDF file , list of non-failure errors
                                Failure returns tuple: None, list of failure errors
    convert_PyPDF_date(orig_date): Pass pypdf2 date, it will return excel style YYYY-MM-DD or original pypdf2 date
    extract_text_local(pdf_path): pass it a file path to a local pdf file and it returns the extracted text (string)
    create_bin_hash(input_io): given binary of pdf file, returns hash of file
    create_text_hash(input_text): given string, returns hash of string
    get_bin_file_size(input_io): given binary of pdf file, returns size in bytes
    extract_text_content_and_pages(pdf): given pypdf2 PDF object, returns tuple of text and page count
    get_bin_file_text(input_io): given binary of pdf file, returns text from that pdf file
"""

import pymongo
import re
import nltk
import sklearn
import urllib2
import PyPDF2
import StringIO
import textract
import hashlib
import sys


def mongo_connect():
    connection = pymongo.MongoClient()
    db = connection.bsa_files
    collection = db.bsa_files
    return (collection)


def tokenize(text):
    extra_stop_words = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
                        't', 'u', 'v', 'w', 'x', 'y', 'z']

    my_stopwords = nltk.corpus.stopwords.words('english') + nltk.corpus.stopwords.words('spanish') + extra_stop_words

    text = re.sub(r"[^\w\s]", '', text)
    text = re.sub(r"[0-9_]", '', text)

    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []

    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search("[a-zA-Z]", token) and token not in my_stopwords and len(
                token) < 30:  # 30 = len(longest english word)
            filtered_tokens.append(token)

    return filtered_tokens


def vectorize_single(document_text):
    collection = mongo_connect()
    tfidf = sklearn.feature_extraction.text.TfidfVectorizer(tokenizer=tokenize,
                                                            vocabulary=collection.find_one(
                                                                {"Document_Class": "english_resources"})[
                                                                "TFIDF_Vocabulary"])
    result = tfidf.transform([document_text])
    return (result)


def download_pdf_from_url(url):
    errors = []
    hdr = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
        'Accept-Encoding': 'none',
        'Accept-Language': 'en-US,en;q=0.8',
        'Connection': 'keep-alive'}

    try:
        temp_req = urllib2.Request(url, headers=hdr)
        try:
            temp_page = urllib2.urlopen(temp_req)
        except urllib2.HTTPError, e:
            return None, [e.fp.read()]

        temp_pdf = PyPDF2.PdfFileReader(StringIO.StringIO(temp_page.read()))

        if temp_pdf.isEncrypted:
            try:
                temp_pdf.decrypt('')
            except:
                errors.append("Attempted decryption failed")

        merger = PyPDF2.PdfFileMerger()

        try:
            merger.append(temp_pdf)
        except:
            errors.append("No file content found")

        doc_info_len = 0
        try:
            doc_info_len = len(temp_pdf.documentInfo)
        except:
            errors.append("Could not Decrypt Metadata on File")

        if doc_info_len > 0:
            try:
                merger.addMetadata(temp_pdf.documentInfo)
            except:
                errors.append("Metadata not written to file")
        temp_return = StringIO.StringIO()
        merger.write(temp_return)
        return temp_return.getvalue(), errors

    except:
        return None,errors


def convert_PyPDF_date(orig_date):
    if len(orig_date) == 23:
        year = orig_date[2:6]
        mon = orig_date[6:8]
        day = orig_date[8:10]
        return (year+"-"+mon+"-"+day)
    return orig_date


def extract_text_local(pdf_path):
    content = textract.process(pdf_path)
    return content


def create_bin_hash(input_io):
    return hashlib.sha1(input_io).hexdigest()


def create_text_hash(input_text):
    return hashlib.sha1(input_text).hexdigest()


def get_bin_file_size(input_io):
    return sys.getsizeof(input_io)


def extract_text_content_and_pages(pdf):
    content = ""
    # iterate pages
    for i in range(0, pdf.getNumPages()):
        # extract the text from each page
        try: content += pdf.getPage(i).extractText() + " \n"
        except: continue
    # collapse whitespaces
    content = " ".join(content.replace("\xa0", " ").split()).encode('utf-8')
    return content,pdf.getNumPages()


def get_bin_file_text(input_io):
    pdf_in_mem = PyPDF2.PdfFileReader(StringIO.StringIO(input_io))
    return extract_text_content_and_pages(pdf_in_mem)[0]