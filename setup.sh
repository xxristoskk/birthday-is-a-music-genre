mkdir -p ~/.streamlit/echo "\
[general]\n\
email = \"xristos.kvtsvros@gmail.com\"\n\
" > ~/.streamlit/credentials.tomlecho "\
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
