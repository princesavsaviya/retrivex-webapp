# searcher/management/commands/start_es.py
import subprocess
import time
import requests
from django.core.management.base import BaseCommand

class Command(BaseCommand):
    help = "Starts Elasticsearch if it's not running"

    def handle(self, *args, **options):
        es_url = "http://localhost:9200"
        try:
            response = requests.get(es_url)
            if response.status_code == 200:
                self.stdout.write(self.style.SUCCESS("Elasticsearch is already running."))
                return
        except requests.exceptions.ConnectionError:
            self.stdout.write("Elasticsearch is not running. Starting it...")

        # Example command; adjust path if needed:
        subprocess.Popen([r"D:\Masters Project\RetriveX Webapp\elasticsearch-8.17.3-windows-x86_64\elasticsearch-8.17.3\bin\elasticsearch.bat"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Wait for ES to start
        for _ in range(10):
            try:
                response = requests.get(es_url)
                if response.status_code == 200:
                    self.stdout.write(self.style.SUCCESS("Elasticsearch started successfully."))
                    return
            except:
                time.sleep(2)
        self.stdout.write(self.style.ERROR("Failed to start Elasticsearch."))
