function millis = getMillis()
    t = clock();
    millis = time()+t(6)*1000;
endfunction