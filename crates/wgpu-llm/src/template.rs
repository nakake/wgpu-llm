use std::collections::HashMap;

pub fn render(template: &str, vars: &HashMap<&str, String>) -> String {
    let mut result = template.to_string();

    for (key, value) in vars {
        let placeholder = format!("{}{}{}", "{{", key, "}}");
        result = result.replace(&placeholder, value);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render() {
        let template = "{{test}}";
        let mut vars = HashMap::new();
        vars.insert("test", "replace".to_string());

        let result = render(template, &vars);

        assert_eq!(result, "replace");
    }
}
