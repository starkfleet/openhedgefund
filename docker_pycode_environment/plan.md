# python snad box plan

we are making a docker based snadbox environment for this
[[Dockerfile]]
docker build -t py_sandbox .

docker run --rm -v "$(pwd)/test_script.py:/test_script.py" py_sandbox python /test_script.py

