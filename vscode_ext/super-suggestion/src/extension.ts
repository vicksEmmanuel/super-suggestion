/** @format */

import * as vscode from 'vscode';

export function activate(context: vscode.ExtensionContext) {
	console.log(
		'Congratulations, your extension "super-suggestion" is now active!'
	);

	// Register the code suggestion provider
	const inlineCompletionProvider: vscode.InlineCompletionItemProvider = {
		provideInlineCompletionItems(
			document: vscode.TextDocument,
			position: vscode.Position,
			context: vscode.InlineCompletionContext,
			token: vscode.CancellationToken
		): vscode.ProviderResult<vscode.InlineCompletionItem[]> {
			const suggestion = new vscode.InlineCompletionItem('= 2 + 2');
			suggestion.range = new vscode.Range(position, position);

			// Set the hint text for the suggestion
			const hintText = '= 2 + 2';
			suggestion.command = {
				command: 'inlineCompletionAccept',
				title: 'Accept',
				arguments: [suggestion.insertText],
			};

			return [suggestion];
		},
	};

	const inlineCompletionDisposable =
		vscode.languages.registerInlineCompletionItemProvider(
			{ pattern: '**' },
			inlineCompletionProvider
		);

	// Set the completion trigger to auto-trigger on every character
	vscode.languages.setLanguageConfiguration('*', {
		wordPattern:
			/(-?\d*\.\d\w*)|([^\`\~\!\@\#\$\%\^\&\*\(\)\-\=\+\[\{\]\}\\\|\;\:\'\"\,\.\<\>\/\?\s]+)/g,
		onEnterRules: [
			{
				beforeText: /.*/,
				action: { indentAction: vscode.IndentAction.None, appendText: '' },
			},
		],
	});

	// Register the command for prompting user input
	const promptCommand = vscode.commands.registerCommand(
		'extension.promptInput',
		() => {
			const editor = vscode.window.activeTextEditor;
			if (editor) {
				const position = editor.selection.active;
				const options = {
					prompt: 'Enter a prompt:',
					placeHolder: 'Type your prompt here',
					value: '',
					buttons: ['Continue', 'Cancel'],
				};

				vscode.window.showInputBox(options).then((value) => {
					if (value === 'Continue') {
						// Perform actions when "Continue" is clicked
						console.log('User clicked Continue');
					}
				});
			}
		}
	);

	// Register the command for uploading the current file
	const uploadCommand = vscode.commands.registerCommand(
		'extension.uploadFolder',
		async () => {
			const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
			// if (workspaceFolder) {
			// 	const folderPath = workspaceFolder.uri.fsPath;

			// 	try {
			// 		const files = await vscode.workspace.fs.readDirectory(
			// 			workspaceFolder.uri
			// 		);

			// 		const uploadPromises = files.map(async ([fileName, fileType]) => {
			// 			if (fileType === vscode.FileType.File) {
			// 				const fileUri = vscode.Uri.joinPath(
			// 					workspaceFolder.uri,
			// 					fileName
			// 				);
			// 				const fileContent = await vscode.workspace.fs.readFile(fileUri);
			// 				const filePath = fileUri.fsPath;

			// 				await axios.post('https://your-upload-service-url', {
			// 					filePath: filePath,
			// 					content: fileContent.toString(),
			// 				});

			// 				console.log(`File uploaded successfully: ${filePath}`);
			// 			}
			// 		});

			// 		await Promise.all(uploadPromises);
			// 		console.log('Folder uploaded successfully');
			// 	} catch (error) {
			// 		console.error('Error uploading folder:', error);
			// 	}
			// } else {
			// 	console.log('No workspace folder is opened');
			// }
		}
	);

	context.subscriptions.push(
		inlineCompletionDisposable,
		promptCommand,
		uploadCommand
	);
}

export function deactivate() {}
