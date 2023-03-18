# List of parameters
#$param_list = "AM AR AROC CMRE EGY EPR EPD ET FLNG FTCO HDSN GSL GLNG GOOG JXN KNTK IBM INSW MP MPW MSFT MPLX TSLA TRTN OXY V VET ZIM"
$param_list = "BBW"


# Splitting the string
$param_array = $param_list -split " "

# Loop through the list
foreach ($param in $param_array) {
    # Call the Python program with the current parameter
    & C:/Users/derek/AppData/Local/Microsoft/WindowsApps/python3.9.exe due_diligence.py $param
}

git commit -am 'daily update' 
git push origin main