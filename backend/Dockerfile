FROM public.ecr.aws/lambda/python:3.10

# Copy the dependencies file
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install the dependencies
RUN pip install -r requirements.txt

# Copy the content of the local src directory to the working directory
COPY model.pkl ${LAMBDA_TASK_ROOT}
COPY scaler.pkl ${LAMBDA_TASK_ROOT}
COPY app.py ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler
CMD [ "app.handler" ]
