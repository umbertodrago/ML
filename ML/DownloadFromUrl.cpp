#include "DownloadFromUrl.hpp"

DownloadFromUrl::DownloadFromUrl(){ }

DownloadFromUrl::~DownloadFromUrl(){ }

void DownloadFromUrl::download(const char *src, const char *dst){
    CURL *curl;
    FILE *fp;
    CURLcode res;
    curl = curl_easy_init();
    
    cout << "Dowloading from " << src << endl;
    
    if (curl) {
        fp = fopen(dst,"wb");
        curl_easy_setopt(curl, CURLOPT_URL, src);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data);
        res = curl_easy_perform(curl);
        curl_easy_cleanup(curl);
        fclose(fp);
        
        string cmd = "gunzip ";
        string sys_cmd =  cmd + dst;
        system(sys_cmd.c_str());
    }
    
    cout << "Done" << endl;
}

size_t DownloadFromUrl::write_data(void *ptr, size_t size, size_t nmemb, FILE *stream) {
    size_t written = fwrite(ptr, size, nmemb, stream);
    return written;
};

