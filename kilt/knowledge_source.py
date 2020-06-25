from pymongo import MongoClient
import requests
from urllib.parse import unquote
import urllib.request
from bs4 import BeautifulSoup
import urllib.parse as urlparse
from urllib.parse import parse_qs
import time
from datetime import datetime
from pymongo import UpdateOne
import pickle

MONGO_CONNECTION_STRING = "mongodb://100.97.69.169:27017/admin"
WIKIPEDIA_TITLE2ID = "/checkpoint/fabiopetroni/KILT/Wikipedia_title2id.p"

dump_date = datetime(2019, 8, 1, 12, 0, 0)  # year, month, day, hour, minute, second.


def _get_original_rendered_wikipedia_page(title_html):

    revid = None
    timestamp = None
    pageid = None
    parentid = None
    pre_dump = False

    url = "https://en.wikipedia.org/w/api.php?action=query&format=json&prop=revisions&rvlimit=500&titles={}&rvlimit=500&rvprop=timestamp|ids".format(
        title_html
    )

    json_data = None

    # limit the total requests (read and write requests together) to no more than 10/minute.

    for retry in range(1, 4):
        try:
            # Package the request, send the request and catch the response: r
            r = requests.get(url)

            # Decode the JSON data into a dictionary: json_data
            json_data = r.json()
            break
        except Exception as e:
            print(url)
            print("Exception A: {}".format(e))
            print()
            time.sleep(2 ** retry)

    if json_data:
        try:
            for x in json_data["query"]["pages"].keys():
                record = json_data["query"]["pages"][x]
                pageid = record["pageid"]
                title = record["title"]
                for revision in record["revisions"]:
                    revid = revision["revid"]
                    parentid = revision["parentid"]
                    timestamp = revision["timestamp"]

                    revision_date = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")

                    if revision_date < dump_date:
                        pre_dump = True
                        break
        except Exception as e:
            print(json_data)
            print(json_data.keys())
            print(url)
            print("Exception B: {}".format(e))
            print()

    return revid, timestamp, pageid, parentid, pre_dump


def _get_pageid_from_api(title, client=None):
    pageid = None

    # 1. try to load from db
    if client == None:
        client = MongoClient(MONGO_CONNECTION_STRING)
    db = client.kilt.title2id
    element = db.find_one({"title": str(title)})
    # print("from mongo:", element)

    # 2. retrieve and store in db
    if element == None:
        title_html = title.strip().replace(" ", "%20")
        url = "https://en.wikipedia.org/w/api.php?action=query&titles={}&format=json".format(
            title_html
        )

        try:
            # Package the request, send the request and catch the response: r
            r = requests.get(url)

            # Decode the JSON data into a dictionary: json_data
            json_data = r.json()

            if len(json_data["query"]["pages"]) > 1:
                print("WARNING: more than one result returned from wikipedia api")

            for _, v in json_data["query"]["pages"].items():
                pageid = v["pageid"]

            db.insert_one({"title": str(title), "pageid": str(pageid)})
        except Exception as e:
            #  print("Exception: {}".format(e))
            pass
    else:
        pageid = element["pageid"]

    return pageid


def _read_url(url):
    with urllib.request.urlopen(url) as response:
        html = response.read()
        soup = BeautifulSoup(html, features="html.parser")
        title = soup.title.string.replace(" - Wikipedia", "").strip()
    return title


def _get_title_from_wikipedia_url(url, client=None):
    title = None

    # 1. try to load from db
    if client == None:
        client = MongoClient(MONGO_CONNECTION_STRING)
    db = client.kilt.url2title
    element = db.find_one({"url": str(url)})
    # print("from mongo:", element)

    # 2. retrieve and store in db
    if element == None:
        title = None
        try:
            title = _read_url(url)
        except Exception:
            try:
                # try adding https
                title = _read_url("https://" + url)
            except Exception:
                #  print("Exception: {}".format(e))
                pass
        if title:
            db.insert_one({"url": str(url), "title": str(title)})
    else:
        title = element["title"]

    return title


class KnowledgeSource:
    def __init__(self):
        self.client = MongoClient(MONGO_CONNECTION_STRING)
        self.db = self.client.kilt.knowledgesource

        self.Wikipedia_title2id = None
        try:
            self.Wikipedia_title2id = pickle.load(open(WIKIPEDIA_TITLE2ID, "rb"))
        except:
            print("Unable to load Wikipedia title to id mapping.")

    def get_all_pages_cursor(self):
        cursor = self.db.find({})
        return cursor

    def get_num_pages(self):
        return self.db.count()

    def get_page_by_id(self, wikipedia_id, retrieve_original_page_if_missing=False):

        page = self.db.find_one({"_id": str(wikipedia_id)})

        if retrieve_original_page_if_missing and (
            "history" not in page or page["history"]["revid"] == None
        ):

            title_html = page["wikipedia_title"].strip().replace(" ", "%20")

            (
                revid,
                timestamp,
                pageid,
                parentid,
                pre_dump,
            ) = _get_original_rendered_wikipedia_page(title_html)

            # little validation
            if pageid and pageid is not None:
                if str(pageid) != str(page["wikipedia_id"]):
                    print(
                        "WARNING pageid",
                        pageid,
                        "!= wikipedia_id",
                        page["wikipedia_id"],
                    )

            page["history"] = {
                "revid": revid,
                "timestamp": timestamp,
                "parentid": parentid,
                "pre_dump": pre_dump,
                "pageid": pageid,
                "url": "https://en.wikipedia.org/w/index.php?title={}&oldid={}".format(
                    title_html, revid
                ),
            }

            # update mongodb
            operations = [UpdateOne({"_id": page["wikipedia_id"]}, {"$set": page})]
            result = self.db.bulk_write(operations)

        return page

    def get_page_by_title(self, wikipedia_title, attempt=0):

        if self.Wikipedia_title2id and (wikipedia_title in self.Wikipedia_title2id):
            wikipedia_id = self.Wikipedia_title2id[wikipedia_title]
            page = self.get_page_by_id(wikipedia_id)
        else:
            page = self.db.find_one({"wikipedia_title": str(wikipedia_title)})
            if not page:
                # try to get the page from URL
                url = "https://en.wikipedia.org/wiki/{}".format(
                    wikipedia_title.replace(" ", "_").strip()
                )
                page = self.get_page_from_url(url, attempt + 1)
        return page

    def get_pages_by_title(self, wikipedia_title):
        page = self.get_page_by_title(wikipedia_title)
        if not page:
            pages = []
        else:
            pages = [page]
        return pages

    def fuzzy_text_search(self, query, limit=None):
        results = list(
            self.client.kilt.ks.find(
                {"$text": {"$search": str(query)}}, {"score": {"$meta": "textScore"}},
            )
        )
        result = sorted(results, key=lambda i: i["score"], reverse=True)
        if limit and len(result) > limit and limit > 0:
            result = result[:limit]
        return result

    def get_page_from_url(self, url, attempt=0):
        page = None

        # 1. try to look for title in the url
        parsed = urlparse.urlparse(url)
        record = parse_qs(parsed.query)
        if "title" in record and attempt < 3:
            title = record["title"][0].replace("_", " ")
            page = self.get_page_by_title(title, attempt + 1)

        # 2. try another way to look for title in the url
        if page == None and attempt < 3:
            title = url.split("/")[-1].replace("_", " ")
            page = self.get_page_by_title(title, attempt + 1)

        # 3: try to retrieve the current wikipedia_id from the url
        if page == None:
            title = _get_title_from_wikipedia_url(url, client=self.client)
            if title:
                pageid = _get_pageid_from_api(title, client=self.client)
                if pageid:
                    page = self.get_page_by_id(pageid)

        return page
