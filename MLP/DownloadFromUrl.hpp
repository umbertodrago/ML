#ifndef DownloadFromUrl_hpp
#define DownloadFromUrl_hpp

#include <stdio.h>
#include <curl/curl.h>
#include <iostream>
#include <string>

using namespace std;

class DownloadFromUrl {
private:
    char *src_url_, *dst_;
    static size_t write_data(void *ptr, size_t size, size_t nmemb, FILE *stream);
    
public:
    DownloadFromUrl();
    ~DownloadFromUrl();
    
    void download(const char *src, const char *dst);
};


#endif /* DownloadFromUrl_hpp */
