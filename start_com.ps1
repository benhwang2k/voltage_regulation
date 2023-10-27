Start-Job -ScriptBlock {
	cd \Users\15053\Desktop\Projects\voltage_regulation
	./.venv/Scripts/Activate.ps1
	python communication.py 0
}
Start-Job -ScriptBlock {
	cd \Users\15053\Desktop\Projects\voltage_regulation
	./.venv/Scripts/Activate.ps1
	python communication.py 1
}
Start-Job -ScriptBlock {
	cd \Users\15053\Desktop\Projects\voltage_regulation
	./.venv/Scripts/Activate.ps1
	python communication.py 2
}
Start-Job -ScriptBlock {
	cd \Users\15053\Desktop\Projects\voltage_regulation
	./.venv/Scripts/Activate.ps1
	python communication.py 3
}
Start-Job -ScriptBlock {
	cd \Users\15053\Desktop\Projects\voltage_regulation
	./.venv/Scripts/Activate.ps1
	python communication.py 4
}
Start-Job -ScriptBlock {
	cd \Users\15053\Desktop\Projects\voltage_regulation
	./.venv/Scripts/Activate.ps1
	python communication.py 5
}

